**Diabetes Prediction Project**

**Overview**
This project aims to predict whether a patient has diabetes based on health metrics using machine learning. The dataset used is the Pima Indians Diabetes Dataset, which includes various health metrics and diabetes outcomes.
Project Steps

**1. Data Loading and Inspection**

    Loaded the dataset and inspected the first few rows.
    Checked data types and missing values.

**2. Exploratory Data Analysis (EDA)**

    Generated summary statistics.
    Visualized distributions using histograms and box plots.
    Computed correlations between features.

**3. Feature Engineering**

    Created new features like BMI and glucose categories.
    Applied log transformation to skewed features.
    Standardized features to have zero mean and unit variance.

**4. Handling Outliers**

    Applied Winsorization to cap extreme values.
    Used visualization to verify the impact of outlier handling.

**5. Model Training and Evaluation**

    Split the data into training and testing sets.
    Trained a logistic regression model and tuned hyperparameters.
    Evaluated model performance using accuracy, precision, recall, F1-score, and ROC-AUC.

**6. Trying Different Algorithms**

    Experimented with Random Forest, SVM, and Gradient Boosting.
    Compared model performance metrics.

**7. Model Deployment**

    Saved the best-performing model.
    Created a Flask API for real-time predictions.
    Discussed deployment strategies and real-world integration.

**Results**

    Best Model: Tuned Logistic Regression
        Accuracy: 74.68%
        ROC-AUC: 0.8250
        Precision/Recall: Balanced performance across classes.

**How to Use**

    Clone the Repository:
    bash

**Copy**

git clone https://github.com/eriqneo/Predictive-Analysis-Diabetes/blob/main/DiabetesPredictiveAnalysis.ipynb

cd diabetes-prediction

Install Dependencies:
bash
Copy

pip install -r requirements.txt

Run the Flask API:
bash
Copy

python app.py

**Make a Prediction:**
Use the following example to send a prediction request:
Python

    Copy

    import requests
    import json

    patient_data = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }

    response = requests.post('http://localhost:5000/predict', json=patient_data)
    print(response.json())

**Dependencies**

    Python 3.11
    Pandas
    NumPy
    Scikit-learn
    Matplotlib
    Seaborn
    Flask
    XGBoost (optional)

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    Pima Indians Diabetes Dataset from the UCI Machine Learning Repository.
    Inspiration from various machine learning tutorials and guides.
