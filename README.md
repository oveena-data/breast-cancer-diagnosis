# 🧬 Cell Image Classification for Breast Cancer Diagnosis

## Project Overview

This project applies popular **machine learning techniques** to the problem of **classifying data from histological cell images** for diagnosing malignant breast cancer. It follows a **best-practice ML workflow** and simulates a **real-world client scenario**, where the client seeks a reliable and interpretable automated classification system to assist in medical diagnostics.

---

## ACS CBOK Alignment 
<img src="https://github.com/user-attachments/assets/27569f12-ea02-4c97-9a1e-2b250064b6e9" alt="Project Logo" height="40">

This project addresses the following Australian Computer Society (ACS) CBOK areas:

- **Abstraction**: Representing the real-world diagnostic problem using models
- **Design**: Developing a scalable and modular ML pipeline
- **Hardware and Software**: Python, Scikit-learn, Pandas, Matplotlib, Jupyter Notebook
- **Data and Information**: Understanding, preparing, and modeling biomedical image data
- **HCI**: Considering interpretability and transparency of ML output for end-users (clinicians)
- **Programming**: Using Python and relevant ML libraries for implementation

---

## Process:

The project follows a multi-step process to:
1. **Preprocess data** (feature selection, scaling...)
2. **Train models** (including k-Nearest Neighbors, Decision Trees, Support Vector Machines, and Stochastic Gradient Descent).
3. **Tune hyperparameters** using GridSearchCV.
4. **Evaluate model performance** using cross-validation and various metrics (accuracy, recall, precision, F1-score, AUC).
5. **Optimise models** by testing performance with selected features.
6. **Final model evaluation** on an unseen test set.
7. **Predict sample class** and provide confidence scores for malignancy predictions.

## Steps and Explanation

### 1. Data Preprocessing
- **Feature Selection:** Initially, the best-performing features were selected based on metrics like T-scores for differentiating benign and malignant samples.
- **Data Splitting:** The dataset was split into training, validation, and test sets.
- **Scaling and Transformation:** Feature scaling and other transformations were applied using pipelines to ensure consistency in preprocessing steps.

### 2. Model Training and Hyperparameter Tuning
Four machine learning models were trained:
- **k-Nearest Neighbors (KNN)**
- **Decision Tree Classifier (DT)**
- **Support Vector Classifier (SVC)**
- **Stochastic Gradient Descent (SGD)**

GridSearchCV was used to tune the hyperparameters of each model.

### 3. Model Evaluation
For each model, performance was evaluated using the following metrics:
- **Accuracy**
- **Recall**
- **Precision**
- **F1-score**
- **AUC** (Area Under the Curve)

The model with the highest AUC score was chosen as the best-performing model.

### 4. Second Round: Excluding Worst Features
In the second round, a smaller set of features was used, excluding the "worst" features based on their T-scores. The performance was again evaluated using the same metrics.

### 5. Final Evaluation
The best model was retrained using a final training set (excluding the worst features) and evaluated on a test set. Performance was measured using AUC score, and the final chosen model was compared between the first and second rounds.

### 6. Final Prediction
A new sample was provided with the features of "worst concave points" and "worst perimeter." The final model was used to predict whether the sample was "benign" or "malignant," and a confidence score for the malignancy class was provided.

---

## Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib

To install the required dependencies, run:

```bash
pip install -r requirements.txt



