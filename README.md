# Diabetes Prediction

This project analyzes a diabetes dataset and builds a machine learning model to predict diabetes status based on health indicators.

## Overview

The notebook:

- Loads and explores a diabetes dataset (`kaggle_diabetes.csv`)
- Visualizes important features by diabetes status
- Handles missing data and cleans the dataset
- Trains a Random Forest Classifier to predict diabetes
- Evaluates the model and demonstrates predictions

## Dataset

The dataset contains 2,000 samples with the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0: Non-Diabetic, 1: Diabetic)

## Exploratory Data Analysis

- Summary statistics and info about data types and missing values
- Visualizations:
  - Age distribution by diabetes status (histogram)
  - Average number of pregnancies by diabetes status (bar plot)
  - Glucose vs BMI scatter plot colored by outcome

## Data Cleaning

- Replaces zeros in key columns with NaN for proper missing value handling
- Fills missing values using means or medians

## Modeling

- Splits the data into features (X) and target (y)
- Uses `train_test_split` to create training and test sets
- Fits a `RandomForestClassifier` on the training data
- Evaluates the classifier with a classification report (precision, recall, f1-score, accuracy)
- Shows the structure of a single decision tree from the forest

## Prediction Demo

- Predicts diabetes status for a sample patient
- Prints the predicted class and probability for each class

## Requirements

- Python 3.13+
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/larabiislem/Diabetes_Prediction.git
   cd Diabetes_Prediction
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not available, manually install:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

4. **Download the dataset:**
   Ensure `kaggle_diabetes.csv` is present in the project directory. If not, you can download it from Kaggle or place your dataset with the same name and columns.

## Running the Project

1. **Open the notebook:**
   ```bash
   jupyter notebook diabet.ipynb
   ```
   or, with JupyterLab:
   ```bash
   jupyter lab diabet.ipynb
   ```

2. **Run all cells step-by-step** to load the data, explore it, train the model, and see the results.

---

Feel free to customize further if you have additional instructions or context!
