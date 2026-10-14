# Viva Voce Examination: Questions & Answers

### Handwritten Digit Recognition Using Deep Learning & Multi-Layer Perceptrons

This guide contains the most frequently asked viva and oral review questions for this project, along with clear, concise, and academically sound answers.

---

### Q1: What is a Multi-Layer Perceptron (MLP), and how does it work in this project?
**Answer:**
A Multi-Layer Perceptron (MLP) is a feedforward artificial neural network consisting of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. In our project:
1. The **input layer** flattens the $28 \times 28$ image into 784 numerical inputs representing pixel intensities.
2. The **hidden layer** (128 neurons with ReLU activation) calculates weighted linear combinations of inputs plus bias, passing them through non-linear activation to learn feature patterns.
3. The **output layer** (10 neurons with Softmax activation) produces probability values for each digit from 0 to 9.

---

### Q2: Why is the Flatten layer required before the Dense layer?
**Answer:**
Dense (fully connected) layers in neural networks perform matrix multiplication with one-dimensional weight vectors ($\mathbf{W} \mathbf{x} + \mathbf{b}$). Because image data naturally arrives as a 2D matrix ($28 \times 28$), the `Flatten` layer unrolls the 2D grid into a single continuous 1D vector of $28 \times 28 = 784$ values without modifying the underlying data.

---

### Q3: Why is the ReLU activation function used in the hidden layer instead of Sigmoid?
**Answer:**
ReLU (Rectified Linear Unit), defined as $f(x) = \max(0, x)$, offers two major advantages over Sigmoid:
1. **Prevents Vanishing Gradients:** Sigmoid saturates at both extremes (approaching 0 or 1), causing derivative values to become extremely small, which stalls backpropagation in deep networks. ReLU has a constant derivative of 1 for all positive inputs.
2. **Computational Efficiency & Sparsity:** ReLU requires only a simple threshold operation (comparison with zero), making training significantly faster while inducing sparse representations (inactive neurons output zero).

---

### Q4: How does the Softmax function work, and why is it used in the output layer?
**Answer:**
Softmax converts raw unconstrained real numbers (logits) into a valid probability distribution. It takes each output $z_i$, computes its exponential $e^{z_i}$, and divides by the sum of exponentials across all 10 classes:
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=0}^{9} e^{z_j}}$$
This guarantees that:
- Every class score is strictly between 0 and 1 ($0 \le p_i \le 1$).
- The sum of all 10 class scores equals exactly 1.0 (100%).
This allows us to interpret the outputs as relative confidence or likelihood scores.

---

### Q5: Why is Sparse Categorical Cross-Entropy loss chosen instead of Mean Squared Error (MSE)?
**Answer:**
Cross-Entropy loss measures the distance between two probability distributions (the true label distribution and the predicted distribution).
- **Steeper Gradients:** When the model makes a confident yet incorrect prediction, Cross-Entropy penalizes it logarithmically ($-\log(p)$), yielding steep gradients that correct weights rapidly. MSE produces flat gradients for confident wrong predictions when coupled with Softmax.
- **Sparse Variant:** "Sparse" categorical cross-entropy allows us to supply true labels as single integer indices (`0, 1, ..., 9`) directly, rather than having to one-hot encode them into 10-element vectors (e.g., `[0, 0, 1, 0, ...]`), which saves memory.

---

### Q6: What is the role of the Adam optimizer, and how does it differ from standard SGD?
**Answer:**
Adam (Adaptive Moment Estimation) is an advanced gradient-based optimization algorithm that combines the benefits of two methods:
- **Momentum:** Accelerates gradient vectors in the right direction by maintaining an exponentially decaying average of past gradients (first moment).
- **RMSprop:** Adapts individual learning rates for each parameter by dividing by the running average of squared gradients (second moment).
Compared to standard Stochastic Gradient Descent (SGD), Adam converges much faster, handles sparse gradients well, and eliminates the need for manual learning rate tuning.

---

### Q7: What is Backpropagation?
**Answer:**
Backpropagation (backward propagation of errors) is the fundamental algorithm used to train neural networks. It operates in two phases:
1. **Forward Pass:** The input image propagates through the network layers to calculate the output probabilities and the overall loss.
2. **Backward Pass:** Using the **Chain Rule of Calculus**, the partial derivatives of the loss with respect to every weight and bias ($\frac{\partial \mathcal{L}}{\partial w}$) are computed from the output layer back to the input layer.
The optimizer then updates the parameters in the direction that minimizes loss ($w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}$).

---

### Q8: What is Overfitting, and how did you prevent it in this project?
**Answer:**
Overfitting occurs when a neural network memorizes noise and specific patterns in the training data rather than generalizing to unseen data. When overfitting happens, training accuracy continues to rise while validation accuracy drops.
In this project, we prevent overfitting by:
1. Keeping the architecture appropriately simple (a single 128-neuron hidden layer rather than hundreds of layers).
2. Splitting the 60,000 training samples into 50,000 training and 10,000 validation images to continuously monitor generalization gap during training.
3. Training for a modest 5 epochs, which achieves high accuracy before memorization can occur.

---

### Q9: Why is the Top-3 Prediction enhancement useful in real-world systems?
**Answer:**
In human handwriting, characters are frequently ambiguous—for example, an open '4' can easily resemble a '9', or a cursive '3' can look like an '8'.
Returning only the top-1 prediction hides this ambiguity from human operators. A **Top-3 Prediction System** provides full visibility by ranking the top 3 candidates with their respective confidence percentages. In automated pipelines (such as bank cheque processing), if the gap between Rank 1 and Rank 2 is small, the system can flag the item for human review.

---

### Q10: What is the purpose of the Confidence Threshold, and does passing it guarantee correctness?
**Answer:**
The Confidence Threshold (e.g., 70%) serves as an automated quality filter. If the highest Softmax score is below this threshold, the system displays a low-confidence warning alerting the user that the input may be blurry, noisy, or outside the training distribution.
**Crucial Point:** Passing the threshold does **not** mathematically guarantee that the prediction is correct. Softmax functions compute relative exponentials; on noisy, inverted, or completely non-digit images, Softmax can still yield high scores for an arbitrary class. Hence, confidence-awareness acts as a practical safeguard, but human verification remains essential for critical systems.
