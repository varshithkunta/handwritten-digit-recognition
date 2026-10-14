"""
Handwritten Digit Recognition Using Deep Learning and Multi-Layer Perceptrons
Model Training and Evaluation Script (train.py)

This script:
1. Loads the MNIST handwritten digit dataset directly from Keras.
2. Preprocesses images (rescaling pixel intensities to [0, 1]).
3. Partitions training data into training (50,000) and validation (10,000) subsets.
4. Builds a Multi-Layer Perceptron (MLP) architecture:
   - Input: 28x28 image
   - Flatten Layer: 784 input nodes
   - Dense Hidden Layer: 128 neurons, ReLU activation
   - Dense Output Layer: 10 neurons, Softmax activation
5. Compiles with Adam optimizer and Sparse Categorical Cross-Entropy loss.
6. Trains for configurable epochs (default: 5) and batch size (default: 128).
7. Evaluates on the unseen 10,000 MNIST test images.
8. Generates evaluation artifacts in `outputs/`:
   - Training & validation accuracy/loss curves
   - Confusion matrix heatmap
   - Classification report (Precision, Recall, F1)
   - Sample actual vs predicted visual grid
   - Metrics summary in JSON
9. Saves the trained model as `model.keras`.
10. Saves sample test digits into `sample_digits/` for interactive testing.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import deep learning framework components
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Scikit-learn for comprehensive evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def create_mlp_model(input_shape=(28, 28), hidden_units=128, num_classes=10):
    """
    Constructs the Multi-Layer Perceptron (MLP) model.
    
    Architecture Breakdown:
    -----------------------
    1. layers.Input(shape=(28, 28)):
       Accepts raw 2D grayscale digit images of dimension 28 rows by 28 columns.
       
    2. layers.Flatten():
       Transforms the 2D grid of 28x28 pixels into a 1D feature vector of 784 inputs.
       This enables feeding pixel intensities directly into dense linear layers.
       
    3. layers.Dense(128, activation='relu'):
       Fully-connected hidden layer with 128 artificial neurons.
       Activation: Rectified Linear Unit (ReLU), f(x) = max(0, x).
       Purpose: Introduces non-linearity so the network can learn non-linear decision boundaries
       and detect complex combinations of pixel patterns without vanishing gradient problems.
       
    4. layers.Dense(10, activation='softmax'):
       Output layer with 10 neurons corresponding to digit classes 0 through 9.
       Activation: Softmax converts raw output logits into a probability distribution
       where each value is in [0, 1] and all 10 outputs sum to 1.0.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape, name="input_layer"),
        layers.Flatten(name="flatten_layer"),
        layers.Dense(hidden_units, activation="relu", name="hidden_layer_relu"),
        layers.Dense(num_classes, activation="softmax", name="output_layer_softmax")
    ], name="MNIST_MLP_Classifier")
    
    return model


def plot_training_history(history, output_path="outputs/training_history.png"):
    """
    Plots training vs validation accuracy and loss over all training epochs.
    """
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], marker='o', linewidth=2, label='Training Accuracy', color='#1f77b4')
    plt.plot(epochs_range, history.history['val_accuracy'], marker='s', linewidth=2, label='Validation Accuracy', color='#2ca02c')
    plt.title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.ylim([0.85, 1.0])
    
    # 2. Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], marker='o', linewidth=2, label='Training Loss', color='#d62728')
    plt.plot(epochs_range, history.history['val_loss'], marker='s', linewidth=2, label='Validation Loss', color='#ff7f0e')
    plt.title('Training and Validation Loss', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss (Cross-Entropy)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved training history plot to: {output_path}")


def plot_confusion_matrix(cm, class_names, output_path="outputs/confusion_matrix.png"):
    """
    Plots a visually clear confusion matrix heatmap for all 10 digit classes.
    """
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - MNIST Test Set', fontsize=14, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)
    
    # Annotate counts inside each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:,}", horizontalalignment="center",
                     verticalalignment="center", color=color, fontsize=9)
            
    plt.ylabel('True Digit Label', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Digit Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved confusion matrix to: {output_path}")


def plot_sample_predictions(x_test, y_test, y_pred, y_probs, output_path="outputs/sample_predictions.png", n_samples=15):
    """
    Plots a grid of test digits showing Actual vs Predicted labels and Softmax confidence score.
    """
    indices = np.random.choice(len(x_test), n_samples, replace=False)
    rows = 3
    cols = 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(13, 8))
    fig.suptitle('Sample Test Predictions (Actual vs Predicted & Confidence)', fontsize=14, fontweight='bold')
    
    for idx, ax in zip(indices, axes.flat):
        img = x_test[idx]
        actual = y_test[idx]
        predicted = y_pred[idx]
        confidence = y_probs[idx][predicted] * 100.0
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        is_correct = (actual == predicted)
        title_color = '#008000' if is_correct else '#cc0000'
        ax.set_title(f"Act: {actual} | Pred: {predicted}\nConf: {confidence:.1f}%",
                     color=title_color, fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved sample predictions to: {output_path}")


def export_sample_test_digits(x_test, y_test, export_dir="sample_digits", count_per_digit=1):
    """
    Exports a clean collection of sample digit images (0-9) as PNG files.
    Allows immediate, zero-friction testing in the Streamlit application.
    """
    os.makedirs(export_dir, exist_ok=True)
    saved = {d: 0 for d in range(10)}
    
    for img_arr, label in zip(x_test, y_test):
        if saved[label] < count_per_digit:
            filename = f"digit_{label}_sample{saved[label] + 1}.png"
            filepath = os.path.join(export_dir, filename)
            # Re-scale back to 0-255 uint8 for standard PNG image
            uint8_img = np.uint8(img_arr * 255.0)
            Image.fromarray(uint8_img).save(filepath)
            saved[label] += 1
        if all(c >= count_per_digit for c in saved.values()):
            break
            
    print(f"[OK] Exported {len(saved) * count_per_digit} sample test digit images to: {export_dir}/")


def train_and_evaluate(epochs=5, batch_size=128, model_path="model.keras", output_dir="outputs"):
    """
    Executes the complete pipeline: dataset loading, preprocessing,
    model construction, training, test evaluation, and artifact generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION (MLP & DEEP LEARNING)")
    print("=" * 60)
    
    # 1. Load MNIST Dataset
    print("\n[Step 1/6] Loading MNIST dataset from Keras...")
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"  - Full Training set: {x_train_full.shape[0]} images, shape: {x_train_full.shape[1:]}")
    print(f"  - Test set:          {x_test.shape[0]} images, shape: {x_test.shape[1:]}")
    
    # 2. Data Preprocessing & Validation Split
    print("\n[Step 2/6] Normalizing pixels from [0, 255] to [0.0, 1.0]...")
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Split 60k training samples into 50k train and 10k validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=10000, random_state=42, stratify=y_train_full
    )
    print(f"  - Training subset:   {x_train.shape[0]} images")
    print(f"  - Validation subset: {x_val.shape[0]} images (reserved for monitoring)")
    print(f"  - Test set:          {x_test.shape[0]} images (strictly for final evaluation)")
    
    # 3. Model Architecture Construction
    print("\n[Step 3/6] Defining Multi-Layer Perceptron (MLP) architecture...")
    model = create_mlp_model(input_shape=(28, 28), hidden_units=128, num_classes=10)
    
    # Compile model
    # - Adam: Adaptive Moment Estimation optimizer
    # - Sparse Categorical Crossentropy: suitable for integer class labels 0..9 without one-hot encoding
    # - Metric: accuracy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # 4. Model Training
    print(f"\n[Step 4/6] Training model for {epochs} epochs (batch size: {batch_size})...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # 5. Model Evaluation on Unseen Test Set
    print("\n[Step 5/6] Evaluating model on 10,000 unseen MNIST test samples...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  - Final Test Loss:     {test_loss:.4f}")
    print(f"  - Final Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Generate predictions on test set for in-depth metrics
    y_test_probs = model.predict(x_test, verbose=0)
    y_test_pred = np.argmax(y_test_probs, axis=1)
    
    # Classification Report
    class_names = [f"Digit {i}" for i in range(10)]
    report_dict = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, y_test_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n" + report_text)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # 6. Save Artifacts & Trained Model
    print("\n[Step 6/6] Saving trained model and evaluation outputs...")
    # Save model in native Keras format
    model.save(model_path)
    print(f"[OK] Saved trained model to: {model_path}")
    
    # Save plots
    plot_training_history(history, os.path.join(output_dir, "training_history.png"))
    plot_confusion_matrix(cm, [str(i) for i in range(10)], os.path.join(output_dir, "confusion_matrix.png"))
    plot_sample_predictions(x_test, y_test, y_test_pred, y_test_probs, os.path.join(output_dir, "sample_predictions.png"))
    
    # Save sample test digits for immediate interactive testing in Streamlit
    export_sample_test_digits(x_test, y_test, export_dir="sample_digits", count_per_digit=2)
    
    # Save metrics JSON for Streamlit to display dynamically
    metrics_summary = {
        "epochs": epochs,
        "batch_size": batch_size,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_pct": f"{test_accuracy * 100:.2f}%",
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[OK] Saved metrics summary to: {metrics_file}")
    
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print(f"You can now launch the web app with: streamlit run app.py")
    print("=" * 60 + "\n")
    
    return model, metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP for Handwritten Digit Recognition")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument("--model_path", type=str, default="model.keras", help="Path to save trained model")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save visual outputs")
    
    args = parser.parse_args()
    train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
