# Diabetes Prediction Using Machine Learning

##  Project Overview

This project uses machine learning classification algorithms to predict whether a patient is likely to have diabetes based on health-related features.

Three machine learning algorithms are applied to the Diabetes dataset:

* Naive Bayes
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

The performance of these algorithms is compared using accuracy, precision, recall, F1-score, and confusion matrix. The best-performing algorithm is identified based on accuracy.

---

##  Objectives

The main objectives of this project are:

1. Select a diabetes dataset.
2. Explore and understand the dataset.
3. Preprocess the data for machine learning.
4. Split the dataset into training and testing data.
5. Apply three machine learning classification algorithms.
6. Evaluate the performance of each algorithm.
7. Compare the algorithms using suitable evaluation metrics.
8. Identify the best-performing algorithm.
9. Predict the diabetes status of a new patient.

---

##  Dataset

The project uses a **Diabetes Dataset** containing **768 records and 9 columns**.

### Dataset Features

| Feature                  | Description                |
| ------------------------ | -------------------------- |
| Pregnancies              | Number of pregnancies      |
| Glucose                  | Glucose concentration      |
| BloodPressure            | Blood pressure             |
| SkinThickness            | Skin thickness             |
| Insulin                  | Insulin level              |
| BMI                      | Body Mass Index            |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age                      | Age of the patient         |
| Outcome                  | Diabetes prediction target |

### Target Variable

**Outcome**

* `0` → No diabetes
* `1` → Diabetes

The dataset contains:

* **768 records**
* **8 input features**
* **1 target variable**
* **No missing values**

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

### Libraries Used

```python
pandas
numpy
scikit-learn
```

### Machine Learning Algorithms

* Gaussian Naive Bayes
* Support Vector Machine with Linear Kernel
* K-Nearest Neighbors with K = 5

---

##  Project Workflow

```text
Diabetes Dataset
       ↓
Data Loading
       ↓
Data Exploration
       ↓
Missing Value Checking
       ↓
Feature and Target Separation
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Machine Learning Models
       ↓
Naive Bayes | SVM | KNN
       ↓
Model Evaluation
       ↓
Performance Comparison
       ↓
Best Algorithm Selection
       ↓
New Patient Prediction
```

---

##  Data Exploration

The dataset was explored using:

* `head()`
* `shape`
* `info()`
* `isnull().sum()`
* `describe()`

### Dataset Shape

```text
(768, 9)
```

### Missing Values

All columns contain **0 missing values**.

Therefore, no missing-value treatment was required.

---

##  Train-Test Split

The dataset was divided into training and testing data using an 80:20 ratio.

```text
Training Data: (614, 8)
Testing Data: (154, 8)
```

The `stratify=y` parameter was used to maintain the class distribution between the training and testing datasets.

---

##  Feature Scaling

`StandardScaler` was used to standardize the input features before training the models.

Feature scaling is especially useful for algorithms such as SVM and KNN because these algorithms are sensitive to the scale of the input features.

---

#  Machine Learning Algorithms

## 1. Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm. It calculates the probability of a patient belonging to each class and predicts the most likely class.

### Result

**Accuracy: 70.78%**

### Classification Report

| Class       | Precision |   Recall | F1-Score |
| ----------- | --------: | -------: | -------: |
| 0           |      0.80 |     0.74 |     0.77 |
| 1           |      0.57 |     0.65 |     0.61 |
| **Overall** |  **0.72** | **0.71** | **0.71** |

### Confusion Matrix

```text
[[74 26]
 [19 35]]
```

---

## 2. Support Vector Machine (SVM)

Support Vector Machine is a supervised machine learning algorithm that finds a decision boundary to separate different classes.

A **linear kernel** was used in this project.

### Result

**Accuracy: 72.08%**

### Classification Report

| Class       | Precision |   Recall | F1-Score |
| ----------- | --------: | -------: | -------: |
| 0           |      0.76 |     0.83 |     0.79 |
| 1           |      0.62 |     0.52 |     0.57 |
| **Overall** |  **0.71** | **0.72** | **0.71** |

### Confusion Matrix

```text
[[83 17]
 [26 28]]
```

---

## 3. K-Nearest Neighbors (KNN)

K-Nearest Neighbors classifies a new data point based on the classes of its nearest neighboring points.

In this project:

```text
K = 5
```

was used.

### Result

**Accuracy: 70.13%**

### Classification Report

| Class       | Precision |   Recall | F1-Score |
| ----------- | --------: | -------: | -------: |
| 0           |      0.75 |     0.80 |     0.78 |
| 1           |      0.58 |     0.52 |     0.55 |
| **Overall** |  **0.69** | **0.70** | **0.70** |

### Confusion Matrix

```text
[[80 20]
 [26 28]]
```

---

#  Performance Comparison

The three algorithms were compared based on their accuracy.

| Algorithm   |   Accuracy |
| ----------- | ---------: |
| Naive Bayes |     70.78% |
| **SVM**     | **72.08%** |
| KNN         |     70.13% |

###  Best Performing Algorithm

**Support Vector Machine (SVM)** achieved the highest accuracy of **72.08%** among the three algorithms tested.

Therefore, based on accuracy, **SVM is the best-performing algorithm for this experiment**.

---

##  New Patient Prediction

A sample new patient was provided to all three trained models.

The input contained the following values:

```text
Pregnancies = 2
Glucose = 120
BloodPressure = 70
SkinThickness = 30
Insulin = 100
BMI = 32
DiabetesPedigreeFunction = 0.5
Age = 35
```

### Predictions

| Algorithm   | Prediction            |
| ----------- | --------------------- |
| Naive Bayes | No diabetes predicted |
| SVM         | No diabetes predicted |
| KNN         | Diabetes predicted    |

The models produced different predictions for the sample patient.

> **Note:** This prediction is only the output of the trained machine learning models and should not be considered a medical diagnosis.

---

##  Warning During Prediction

A warning was displayed:

```text
UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
```

This occurs because the scaler was trained using a pandas DataFrame with column names, while the new patient was supplied as a NumPy array.

The warning does **not** prevent the model from producing a prediction.

---

## Evaluation Metrics

The following metrics were used to evaluate the models:

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many of the samples predicted as a particular class actually belong to that class.

### Recall

Measures how many of the actual samples in a class were correctly identified.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

---

##  Project Structure

```text
Diabetes-ML-Project/
│
├── diabetes_dataset.csv
├── diabetes_prediction.py
├── README.md
└── screenshots/
    ├── dataset_output.png
    ├── naive_bayes_output.png
    ├── svm_output.png
    ├── knn_output.png
    └── prediction_output.png
```

---

## How to Run the Project

### Step 1: Install Required Libraries

Open the terminal and run:

```bash
python -m pip install pandas numpy scikit-learn
```

### Step 2: Place the Dataset

Keep the `diabetes_dataset.csv` file in the project folder.

### Step 3: Run the Python Program

```bash
python diabetes_prediction.py
```

### Step 4: View the Results

The program displays:

* Dataset information
* Dataset statistics
* Missing-value information
* Training and testing data size
* Naive Bayes performance
* SVM performance
* KNN performance
* Confusion matrices
* Best-performing algorithm
* New patient predictions

---

# Conclusion

This project demonstrates the use of supervised machine learning algorithms for diabetes classification.

Three algorithms — **Naive Bayes, SVM, and KNN** — were trained and evaluated on the same dataset.

The obtained accuracies were:

* Naive Bayes: **70.78%**
* SVM: **72.08%**
* KNN: **70.13%**

Among the three algorithms, **SVM achieved the highest accuracy of 72.08%**, making it the best-performing algorithm in this experiment based on accuracy.

The project demonstrates the complete machine learning workflow, from dataset exploration and preprocessing to model training, evaluation, comparison, and prediction.

---

