# Handwritten Digit Recognition Using Deep Learning and Multi-Layer Perceptrons

An end-to-end, academic-grade Machine Learning project for recognizing handwritten digits (0–9) using a Multi-Layer Perceptron (MLP) trained on the MNIST benchmark dataset.

The project features a **Top-3 Prediction System**, **Confidence-Aware Prediction Warning**, **Automated Preprocessing & Polarity Correction**, and an interactive **Streamlit Web Application**.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features & Enhancements](#key-features--enhancements)
3. [Deep Learning Architecture](#deep-learning-architecture)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Workflow & Execution](#workflow--execution)
7. [Streamlit Web Interface](#streamlit-web-interface)
8. [Google Colab Alternative](#google-colab-alternative)
9. [Troubleshooting](#troubleshooting)
10. [Academic Report & Viva Questions](#academic-report--viva-questions)

---

## 1. Project Overview

The objective of this project is to develop and evaluate an artificial neural network capable of classifying grayscale handwritten digits (0 through 9). The model is trained on the standard **MNIST** dataset (60,000 training images, 10,000 test images) using TensorFlow and Keras.

Beyond simple classification, this project incorporates two practical engineering enhancements:
- **Top-3 Prediction System:** Ranks and displays the three most likely digit candidates with their Softmax probability scores.
- **Confidence-Aware Prediction:** Triggers an explicit warning whenever the model's highest Softmax score falls below a user-defined threshold (default 70%).

---

## 2. Key Features & Enhancements

- **Feature A: Digit Classification**
  - Accepts arbitrary digit images, resizes, normalizes, and classifies them into one of the 10 digit classes (0–9).
- **Feature B: Top-3 Prediction System**
  - Displays the 3 highest Softmax class scores with visual probability progress bars.
  - Acknowledges that Softmax scores represent relative model confidence and not absolute calibrated probabilities.
- **Feature C: Confidence-Aware Prediction**
  - Allows users to configure a minimum confidence threshold (default 70%).
  - Banners alert users when predictions are uncertain, preventing blind trust in low-scoring classifications.
- **Feature D: Training & Validation Visualizations**
  - Generates clear plots for training vs. validation accuracy and cross-entropy loss over epochs (`outputs/training_history.png`).
- **Feature E: Model Evaluation**
  - Evaluates performance on the 10,000 unseen MNIST test samples.
  - Exports a confusion matrix heatmap (`outputs/confusion_matrix.png`), actual vs. predicted samples grid (`outputs/sample_predictions.png`), and per-class classification report (Precision, Recall, F1-Score).
- **New Feature: Interactive Edge-Case & Stress Lab**
  - Built-in test gallery for synthetic edge cases (Blank canvas, Inverted polarity, Off-center digits, Ambiguous 4/9 shapes, Non-digit doodles, and Transparent PNGs).
- **New Feature: Full 10-Class Softmax Probability Distribution**
  - Interactive bar chart breaking down output scores across all 10 digit classes simultaneously.
- **New Feature: Preprocessing Diagnostics Inspector**
  - Displays detected polarity, original resolution, bounding box coordinates, and aspect ratio.
- **Automated Image Preprocessing & Polarity Inversion**
  - Inverts real-world dark-ink-on-white-paper images into MNIST-standard white-ink-on-black-canvas.
  - Detects foreground bounding boxes, preserves digit aspect ratio, and centers within a 28×28 canvas.
- **Clean UI & Hidden Deploy Button**
  - Custom CSS removes default Streamlit deploy banners, toolbar headers, and hamburger menus for an unbranded, professional academic aesthetic.

---

## 3. Deep Learning Architecture

The model is a sequential Multi-Layer Perceptron (MLP):

```
Input Image (28×28 Grayscale)
           │
           ▼
┌──────────────────────────┐
│ Flatten Layer            │  --> Reshapes 28×28 image into 784 1D feature vector
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Dense Hidden Layer       │  --> 128 Neurons, ReLU Activation: f(x) = max(0, x)
│ (100,480 parameters)     │  --> Learns non-linear feature representations
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Dense Output Layer       │  --> 10 Neurons, Softmax Activation: σ(z)_i = e^(z_i) / Σ e^(z_j)
│ (1,290 parameters)       │  --> Normalizes logits into probabilities summing to 1.0
└──────────────────────────┘
           │
           ▼
10 Class Probabilities (Digits 0–9)
```

- **Optimizer:** `Adam` (Adaptive Moment Estimation, learning rate = 0.001)
- **Loss Function:** `Sparse Categorical Cross-Entropy` (computes cross-entropy error on integer labels 0..9)
- **Metric:** `Accuracy`
- **Total Trainable Parameters:** 101,770

---

## 4. Project Structure

```text
dla project/
├── train.py                  # Trains MLP, evaluates test set, saves model & plots
├── app.py                    # Streamlit interactive web application
├── utils.py                  # Image preprocessing, Top-3 ranking, and threshold checks
├── requirements.txt          # Python library dependencies
├── model.keras               # Saved trained Keras model (generated by train.py)
├── README.md                 # Complete project documentation
├── PROJECT_REPORT.md         # Full academic project report
├── VIVA_QUESTIONS.md         # Comprehensive viva voce Q&A guide
├── sample_digits/            # Sample digit images exported from test set for quick testing
└── outputs/                  # Evaluation artifacts
    ├── metrics.json          # Numerical evaluation metrics
    ├── training_history.png  # Accuracy & loss learning curves
    ├── confusion_matrix.png  # 10x10 test confusion matrix
    └── sample_predictions.png# Grid of actual vs predicted digits
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.

### Step-by-Step Setup
1. Clone or navigate to the project directory:
   ```bash
   cd "d:/dla project"
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Workflow & Execution

### Step 1: Train the MLP Model
Run `train.py` to download the MNIST dataset, train the network for 5 epochs, evaluate on the 10,000 test set, and export artifacts:
```bash
python train.py
```

Optional arguments:
```bash
python train.py --epochs 5 --batch_size 128
```

This will generate:
- `model.keras` — Saved model file.
- `outputs/metrics.json` — Evaluation summary.
- `outputs/training_history.png` — Accuracy & loss curves.
- `outputs/confusion_matrix.png` — Confusion matrix heatmap.
- `outputs/sample_predictions.png` — Sample prediction grid.
- `sample_digits/*.png` — Pre-extracted test digits for instant testing.

### Step 2: Run the Streamlit Application
Launch the interactive web interface:
```bash
streamlit run app.py
```

The web app will automatically open in your default browser at `http://localhost:8501`.

---

## 7. Streamlit Web Interface

The web app features three structured tabs:
1. **🔮 Interactive Prediction:**
   - Test using pre-packaged sample digits or upload any PNG/JPG image.
   - Live visual verification: shows original image alongside the preprocessed 28×28 canvas.
   - Primary predicted digit with large visual callout.
   - Top-3 candidate predictions with relative confidence bars.
   - Configurable confidence threshold slider (default 70%) with instant warning or pass indicators.
2. **📊 Model Performance & Evaluation:**
   - Real test accuracy and test loss metrics.
   - High-resolution training curves and confusion matrix.
   - Detailed classification report table (Precision, Recall, F1-Score, and Support).
3. **🧠 Architecture & Explanation:**
   - Explains the purpose of each layer, activation function, loss, and optimizer.
   - Compares MLP with Convolutional Neural Networks (CNNs).

---

## 8. Google Colab Alternative

If you do not have a local Python environment or GPU:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `train.py` and `utils.py`.
3. In a Colab cell, execute:
   ```bash
   !pip install -r requirements.txt
   !python train.py
   ```
4. Download the resulting `model.keras` and `outputs/` folder to your local machine to run `app.py`.

---

## 9. Troubleshooting

- **Issue:** `ModuleNotFoundError: No module named 'tensorflow'`
  - **Solution:** Ensure your active Python environment has the requirements installed: `pip install -r requirements.txt`. On Windows, verify you are running `python` or `py` linked to the correct environment.
- **Issue:** Model file `model.keras` not found when running `app.py`
  - **Solution:** Execute `python train.py` first, or click the **"🚀 Train Model Now"** button in the Streamlit sidebar.
- **Issue:** Custom uploaded digit is misclassified
  - **Solution:** Real-world handwriting often has noisy backgrounds or thin strokes. Check the preprocessed 28×28 image preview in Tab 1 to see how the model sees the digit. Use bold, clear strokes on clean paper.

---

## 10. Academic Report & Viva Questions

- Refer to [PROJECT_REPORT.md](file:///d:/dla%20project/PROJECT_REPORT.md) for the complete academic write-up suitable for submission.
- Refer to [VIVA_QUESTIONS.md](file:///d:/dla%20project/VIVA_QUESTIONS.md) for 10 common oral examination questions with answers.
