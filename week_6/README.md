# Introduction to Neural Networks and Deep Learning

Welcome to the **Introduction to Neural Networks and Deep Learning** course! This course is designed to help beginners understand the fundamentals of neural networks, the mathematical basis behind them, and the process of training deep learning models. Let’s dive in!

---

## Table of Contents

1. [Introduction to Neural Networks and Deep Learning](#introduction-to-neural-networks-and-deep-learning)
2. [Understanding the Structure of Neural Networks](#understanding-the-structure-of-neural-networks)
3. [Activation Functions – Adding Non-Linearity](#activation-functions--adding-non-linearity)
4. [Forward Propagation](#forward-propagation)
5. [Loss Functions and Training](#loss-functions-and-training)
6. [Backpropagation – Learning from Mistakes](#backpropagation--learning-from-mistakes)
7. [Training the Neural Network – Putting It All Together](#training-the-neural-network--putting-it-all-together)
8. [Model Performance and Generalization](#model-performance-and-generalization)

---

## 1. Introduction to Neural Networks and Deep Learning

### 1.1 What is Deep Learning?
- **Definition**: Deep learning is a subset of machine learning and artificial intelligence that uses neural networks with multiple layers to model complex patterns in data.
- **Applications**: Includes image recognition, language translation, and self-driving cars.
- **Why Now?**: Advances in computing power, data availability, and algorithms have driven recent deep learning breakthroughs.

### 1.2 Introduction to Neural Networks
- **Basic Idea**: Neural networks are computational models inspired by the human brain, consisting of layers of neurons that adjust parameters through training.
- **Historical Background**: The development of neural networks, from early perceptrons to modern deep learning.

---

## 2. Understanding the Structure of Neural Networks

### 2.1 Neurons
Each neuron performs a simple computation using the equation:

$ z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b $

where:
- \( z \) is the neuron's raw output.
- \( w_i \) is the weight applied to input \( x_i \).
- \( b \) is the bias term.

After calculating \( z \), an **activation function** is applied to introduce non-linearities, allowing the network to learn complex relationships.

### 2.2 Layers in a Neural Network
- **Input Layer**: Receives raw data (e.g., image pixels).
- **Hidden Layers**: Perform computations to extract abstract patterns.
- **Output Layer**: Produces the final output, such as class probabilities or predictions.
- **Depth**: Networks with multiple hidden layers are known as "deep" networks.

### 2.3 Types of Neural Networks
- **Feedforward Neural Networks (FNNs)**: Data flows one-way from input to output.
- **Convolutional Neural Networks (CNNs)**: Specialized for image data.
- **Recurrent Neural Networks (RNNs)**: Handle sequence data, like time series.

---

## 3. Activation Functions – Adding Non-Linearity

### 3.1 The Need for Activation Functions
Activation functions introduce non-linearity, enabling neural networks to model complex relationships beyond simple linear mappings.

### 3.2 Common Activation Functions
- **Sigmoid Function**:
  $
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $
  Maps input to a range between 0 and 1, often used in binary classification.

- **ReLU (Rectified Linear Unit)**:
  $
  f(z) = \max(0, z)
  $
  Commonly used in hidden layers, it outputs 0 for negative inputs and the input itself for positive values.

- **Softmax Function**:
  $
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}
  $
  Normalizes output to represent probabilities across multiple classes, used in multi-class classification.

---

## 4. Forward Propagation

### 4.1 Concept of Forward Propagation
Forward propagation is the process of passing inputs through the network to compute predictions.

### 4.2 Step-by-Step Forward Pass
1. Multiply inputs by weights, add biases at each neuron.
2. Pass the result through an activation function.
3. Continue layer-by-layer until reaching the output layer.

### 4.3 Example Calculation
For a network with two inputs, two layers, and one output, follow the computation step-by-step to illustrate how predictions are generated.

---

## 5. Loss Functions and Training

### 5.1 Purpose of Loss Functions
Loss functions measure how well predictions match the actual target values.

### 5.2 Common Loss Functions
- **Mean Squared Error (MSE)** for regression:
  $
  MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
  $

- **Cross-Entropy Loss** for classification:
  $
  L = -\sum_{i=1}^n y_i \cdot \log(\hat{y}_i)
  $

---

## 6. Backpropagation – Learning from Mistakes

### 6.1 Concept of Backpropagation
Backpropagation adjusts weights based on prediction errors, using calculus (the chain rule) to calculate gradients of the loss function.

### 6.2 Calculating Gradients
- **Partial Derivatives**: Show how small changes in weights affect the loss.
- **Gradient of the Loss Function**: Helps in determining how to adjust each weight to minimize the loss.

### 6.3 Updating Weights Using Gradient Descent
- **Gradient Descent**:
  $
  w = w - \alpha \frac{\partial L}{\partial w}
  $
  where \( \alpha \) is the learning rate, controlling the step size.

- **Learning Rate**: A crucial parameter for efficient learning.
- **Variants**: Introduce Stochastic and Mini-Batch Gradient Descent for large datasets.

---

## 7. Training the Neural Network – Putting It All Together

### 7.1 Steps in Training
1. **Initialize Weights and Biases** randomly.
2. **Forward Propagation** to get predictions.
3. **Compute Loss** to measure prediction accuracy.
4. **Backpropagation** to calculate gradients and update weights.
5. **Repeat** for multiple epochs until the model converges.

### 7.2 Training Hyperparameters
- **Epochs**: Number of complete data passes.
- **Batch Size**: Number of samples before updating.
- **Early Stopping**: Stop training when performance on validation data ceases to improve.

---

## 8. Model Performance and Generalization

### 8.1 Overfitting vs. Underfitting
- **Overfitting**: Model performs well on training data but poorly on new data.
- **Underfitting**: Model is too simple to capture patterns.

### 8.2 Regularization Techniques
- **Dropout**: Randomly drops neurons during training to avoid over-reliance.
- **L2 Regularization**: Penalizes large weights to prevent overfitting.

### 8.3 Model Evaluation Metrics
- **Accuracy**: Common metric for classification.
- **Precision, Recall, F1 Score**: Important for imbalanced data.
- **Confusion Matrix**: Visual representation of classification results.
