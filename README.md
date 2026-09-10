# Diabetes Prediction Using Machine Learning

##  Project Overview

This project applies machine learning classification algorithms to a diabetes dataset to predict whether a patient is likely to have diabetes.

Three machine learning algorithms are implemented and compared:

* Naive Bayes
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

The performance of these algorithms is evaluated using suitable classification metrics, and the best-performing algorithm is identified based on accuracy.



##  Objectives

The main objectives of this project are:

1. Select and understand a diabetes dataset.
2. Preprocess the dataset for machine learning.
3. Split the dataset into training and testing data.
4. Apply three classification algorithms.
5. Evaluate the performance of each algorithm.
6. Compare the results using accuracy, precision, recall, F1-score, and confusion matrix.
7. Identify the best-performing algorithm.
8. Predict the diabetes status of a new patient.



##  Dataset

The project uses a **Diabetes Dataset** containing patient-related health information.

### Target Variable

**Outcome**

* `0` → No Diabetes
* `1` → Diabetes

### Input Features

The dataset contains features such as:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

These features are used to predict the `Outcome`.



##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

### Machine Learning Algorithms

* Gaussian Naive Bayes
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)



##  Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Separate Features and Target
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Apply ML Algorithms
   ↓
Naive Bayes | SVM | KNN
   ↓
Model Evaluation
   ↓
Compare Results
   ↓
Identify Best Algorithm
   ↓
Predict New Patient
```

---

##  Data Preprocessing

The dataset is loaded using Pandas.

The following steps are performed:

1. Display the first few records.
2. Check the shape of the dataset.
3. Check the data information and data types.
4. Check for missing values.
5. Generate descriptive statistics.
6. Separate input features and target variable.
7. Split the data into training and testing sets.
8. Apply `StandardScaler` to scale the input features.

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

Stratified splitting is used to maintain the class distribution in the training and testing datasets.

---

##  Machine Learning Algorithms

### 1. Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm. It predicts the class of a patient based on the probability of the input features.

### 2. Support Vector Machine (SVM)

SVM is a supervised learning algorithm that finds a suitable decision boundary to separate different classes.

A **linear kernel** is used in this project.

### 3. K-Nearest Neighbors (KNN)

KNN classifies a new data point based on the classes of its nearest neighboring data points.

In this project, `k = 5` is used.



##  Evaluation Metrics

The models are evaluated using the following metrics:

### Accuracy

Accuracy represents the percentage of correctly classified predictions.

### Precision

Precision indicates how many of the patients predicted as diabetic are actually diabetic.

### Recall

Recall indicates how many of the actual diabetic patients are correctly identified by the model.

### F1-Score

F1-score provides a balance between precision and recall.

### Confusion Matrix

The confusion matrix shows:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)



##  Results

The three algorithms are compared based on their classification performance.

| Algorithm   |   Accuracy |  Precision |     Recall |   F1-Score |
| ----------- | ---------: | ---------: | ---------: | ---------: |
| Naive Bayes | Add output | Add output | Add output | Add output |
| SVM         | Add output | Add output | Add output | Add output |
| KNN         | Add output | Add output | Add output | Add output |

> **Note:** Replace the values with the actual results obtained when running the program.

### Best Performing Algorithm

The algorithm with the highest accuracy is selected as the **best-performing algorithm**.

The program automatically identifies the best algorithm using:

```python
best_algorithm = results.loc[
    results["Accuracy"].idxmax(), "Algorithm"
]
```



##  New Patient Prediction

The trained models are also used to predict the diabetes status of a new patient.

The prediction can be:

* **Diabetes predicted**
* **No diabetes predicted**

The new patient's data is scaled using the same `StandardScaler` used during model training before making predictions.



## ▶️ How to Run the Project

### Step 1: Install Python

Make sure Python is installed on your system.

### Step 2: Install Required Libraries

Open the terminal and run:

```bash
python -m pip install pandas numpy scikit-learn
```

### Step 3: Place the Dataset

Keep `diabetes_dataset.csv` in the project folder.

### Step 4: Run the Python Program

```bash
python diabetes_prediction.py
```

### Step 5: View the Output

The program displays:

* Dataset information
* Missing-value information
* Training and testing data sizes
* Naive Bayes evaluation
* SVM evaluation
* KNN evaluation
* Accuracy comparison
* Best-performing algorithm
* New patient predictions



##  Conclusion

This project demonstrates the application of supervised machine learning algorithms for diabetes prediction.

Naive Bayes, SVM, and KNN are trained and evaluated on the same dataset. Their performance is compared using accuracy, precision, recall, F1-score, and confusion matrices.

Based on the obtained results, the algorithm with the highest accuracy is considered the **best-performing algorithm** for this dataset.

The project provides a simple demonstration of how machine learning can be used for classification and prediction using healthcare-related data.




