# Model Ensembles & Bagging in Machine Learning

   Welcome to the **Model Ensembles & Bagging in Machine Learning** repository. 
   This project demonstrates the implementation of various boosting methods using the *Titanic Dataset*, providing a comprehensive comparison of their performances.

## Table of Contents

   1. [Introduction](#introduction)
   2. [Dataset](#dataset)
   3. [Boosting Methods](#boosting-methods)
       - [Decision Tree](#boosting-with-decision-tree)
       - [XGBoost](#boosting-with-xgboost)
       - [LightGBM](#boosting-with-lightgbm)
   4. [Installation](#installation)
   5. [Usage](#usage)
   6. [Example Results](#results)
   7. [License](#license)

## Introduction

   Ensemble methods combine multiple base models to improve predictive performance. This repository explores various ensemble techniques applied to the Titanic dataset. 
   
   
   This repository explores:

   - [Bagging with XGBoost](Bagging%20With%20XGboost)
   - [Bagging with LightGBM](Bagging%20With%20lightgbm)
   - [Bagging with Decision Tree](Bagging%20With%20DecisionTree)


   Each method is tested on the Titanic dataset to illustrate their effectiveness and performance differences.

## Dataset

   The [Titanic dataset](titanic.csv) is a famous dataset used in machine learning and statistics for binary classification.
   The goal is to predict whether a passenger survived the Titanic disaster based on features such as age, gender, class, etc.

## Boosting Methods

   * *Bagging with LightGBM*

   Bagging (Bootstrap Aggregating) with LightGBM enhances the performance of LightGBM models by combining predictions from multiple variations of LightGBM trained on different subsets of the training data. Each subset is sampled with replacement (bootstrap sampling), and a separate LightGBM model is trained on each subset. This ensemble technique helps in reducing overfitting and improving the model's ability to generalize on unseen data.

   [Bagging with LightGBM](Bagging%20With%20lightgbm/Bagging_Implement_With_lightgbm.ipynb)


   * *Bagging with XGBoost*

   Bagging (Bootstrap Aggregating) with XGBoost enhances the performance of XGBoost models by combining predictions from multiple variations of XGBoost trained on different subsets of the training data. Each subset is sampled with replacement (bootstrap sampling), and a separate XGBoost model is trained on each subset. This ensemble technique helps in improving the model's robustness and generalization capabilities.

   [Bagging with XGBoost](Ensemble%20With%20Bagging/Bagging_Implement_With_SVM.ipynb)


   * *Bagging with Decision Tree*

    Bagging (Bootstrap Aggregating) is an ensemble technique that aims to improve the stability and accuracy of machine learning algorithms by averaging multiple variations of a single base model trained on different subsets of the training data. Each subset is sampled with replacement (bootstrap sampling), and a separate model is trained on each subset. Bagging helps in reducing variance and can be applied to various base models, including decision trees. Here, we explore bagging with decision trees in the notebook:

    [Bagging with Decision Tree Implementation](Bagging%20With%20DecisionTree/Bagging_Implement_With_DecisionTree.ipynb)


   Each notebook demonstrates the application of bagging techniques to the Titanic dataset, evaluating their performance metrics such as accuracy, precision, recall, and F1-score.
   These methods illustrate how ensemble techniques, specifically bagging, can enhance predictive accuracy and robustness in classification tasks.

## Installation

   To run the scripts and notebooks in this repository, ensure you have Python 3.x and the following libraries installed:

   | Library      | Version |
   |--------------|---------|
   | numpy        | 1.21.0  |
   | pandas       | 1.3.0   |
   | matplotlib   | 3.4.2   |
   | scikit-learn | 0.24.2  |
   | lightgbm     | 3.2.1   |
   | xgboost      | 1.4.0   |


   You can install these dependencies using pip:

   'pip install numpy pandas matplotlib scikit-learn lightgbm xgboost'

## Usage

   1. Clone the repository
   2. Open and run the Jupyter notebooks (*.ipynb) in your preferred environment. Each notebook demonstrates a specific boosting method.

## Example Results

   1. Classification Report Comparison

| Algorithm     | Precision (Not Survived) | Recall (Not Survived) | F1-score (Not Survived) | Precision (Survived) | Recall (Survived) | F1-score (Survived) | Accuracy |
|---------------|--------------------------|-----------------------|--------------------------|----------------------|-------------------|---------------------|----------|
| **LightGBM**   | 0.67                     | 0.57                  | 0.62                     | 0.76                 | 0.83              | 0.79                | 0.73     |
| **XGBoost**    | 0.70                     | 0.50                  | 0.58                     | 0.74                 | 0.87              | 0.80                | 0.73     |
| **Decision Tree** | 0.63                     | 0.86                  | 0.73                     | 0.89                 | 0.70              | 0.78                | 0.76     |

   - **Accuracy:** Overall accuracy of the model on the test set.
   - **Precision:** Ability of the model to avoid false positives.
   - **Recall:** Ability of the model to find all positive instances.
   - **F1-score:** Harmonic mean of precision and recall.

   These results demonstrate the performance metrics for each algorithm based on the evaluation of their classification reports.
   Each algorithm shows varying strengths in different aspects of the classification task, with LightGBM and XGBoost generally achieving higher scores compared to Decision Tree on this dataset.

   2.  ROC

   * The [Receiver Operating Characteristic (ROC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
   The curve is created by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
   * The area under the ROC curve (AUC) provides an aggregate measure of performance across all classification thresholds.
   An AUC of 1.0 represents a perfect model, while an AUC of 0.5 represents a model with no discrimination ability.


   3. Confusion Matrix

   * The [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix) is a table used to describe the performance of a classification model on a set of test data for which the true values are known.
   It shows the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.
   * This matrix helps in understanding the types of errors the model is making and provides insights into the accuracy, precision, recall, and F1-score metrics.

## License

   This repository is licensed under the GNU General Public License (GPL) v3.0.
   See the [LICENSE](./LICENSE) file for more details.