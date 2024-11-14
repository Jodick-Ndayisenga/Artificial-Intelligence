# Week 4: Support Vector Machines and K-Nearest Neighbors

This week, we’ll dive into two popular supervised learning algorithms: **Support Vector Machines (SVM)** and **K-Nearest Neighbors (KNN)**. Both algorithms are frequently used for classification tasks, but they approach the problem of categorizing data in fundamentally different ways.

---

## Topics
- Introduction to Support Vector Machines (SVM) and K-Nearest Neighbors (KNN)

---

## Theory and Concepts

### 1. Support Vector Machines (SVM)

**Support Vector Machines (SVM)** is a supervised learning algorithm that classifies data by identifying the hyperplane that best separates different classes. Its main goal is to maximize the "margin" between classes, ensuring that data points are as far as possible from the decision boundary, reducing classification errors.

- **Maximal Margin Classifier**:
  - The **margin** is defined as the distance between the hyperplane (decision boundary) and the nearest data point from any class.
  - SVM attempts to find the hyperplane that maximizes this margin to reduce the likelihood of misclassification.
  
- **Support Vectors**:
  - **Support vectors** are the data points that lie closest to the decision boundary or hyperplane. These are the critical points in determining the optimal position and orientation of the hyperplane.
  - Only the support vectors affect the margin and decision boundary, making SVM robust against outliers.

- **Kernel Functions**:
  - SVM can be extended to classify non-linear data by using **kernel functions**. Kernels map data into a higher-dimensional space where it becomes linearly separable.
  - Common kernels include:
    - **Linear Kernel**: Suitable for linearly separable data.
    - **Polynomial Kernel**: Useful when the relationships between features are polynomial in nature.
    - **Radial Basis Function (RBF) or Gaussian Kernel**: Effective for complex, non-linear data with localized clusters.

#### Example: Classifying Spam Emails
Imagine classifying emails as "spam" or "not spam" based on the frequency of specific words. SVM would find the optimal boundary in feature space to separate these two classes by identifying the most relevant "support vectors." These support vectors would help define the boundary that best separates "spam" from "not spam."

---

### 2. K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is an instance-based learning algorithm that classifies new data points based on the majority class among their **k** nearest neighbors in the feature space.

- **Distance Metrics**:
  - **Distance metrics** are used to determine which points are closest to each other in the feature space. Common metrics include:
    - **Euclidean Distance**: Measures straight-line distance between two points.
    - **Manhattan Distance**: Measures distance along the axes of the feature space.
  
- **Choosing the Optimal k**:
  - The **k** value represents the number of neighbors to consider for classification. Choosing an optimal **k** is crucial:
    - Small values of **k** make the model sensitive to noise.
    - Large values can cause the model to overlook local patterns.
  - **Cross-validation** is typically used to identify the best value for **k**.

- **Weighted Voting**:
  - In some cases, KNN assigns weights to neighbors based on their proximity to the new data point, giving more influence to closer points in determining the final class label.

#### Example: Predicting Movie Genres
Imagine a system where a new movie's genre is predicted based on features such as keywords, plot length, and actors. KNN would look at a set number of similar movies to classify this new movie. If most of its neighbors belong to the "Action" genre, it’s likely that this new movie will also be classified as "Action."

---

By the end of this week, you'll gain a deep understanding of the theoretical and practical aspects of SVM and KNN, including their core concepts and key parameters. This foundation will enable you to apply these algorithms to real-world datasets and understand the differences in their performance and usage.