# ComparingPerformanceClassificationModels

This project uses classification models and compares their performance using the Accuracy classification score. It aims to identify the most effective model for predicting student performance based on several influencing factors.
It aims to predict whether customers should get a loan based on several influencing factors.

## Dataset

The "Loan" dataset analyzes factors affecting approving a loan. It comprises records for 382 customers with the following details:

### Variables
- **Gender**: Male or Female.
- **Married Status**: Yes or No.
- **Dependents**: Number of dependents.
- **Education**: Applicant's education level.
- **Self_Employed**: Whether the applicant is self-employed or not.
- **Applicant Income**: Income of the applicant.
- **Co-Applicant Income**: Income of the co-applicant.
- **LoanAmount**: Loan amount in thousands.
- **Credit_History**: Credit history meets guidelines.
- **Property_Area**: Urban/ Semi Urban/ Rural.


### Target Variable
- **Loan Status**: Loan approved (Yes or No).

### Data Source
- [Loan Dataset on Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction/code)

## Libraries
- Numpy
- Matplotlib
- Pandas
- Scikit-learn

## Steps in the Practice
1. **Preparing the Dataset**: Handle missing data using the SimpleImputer from the sklearn library.
2. **Splitting the Dataset**: Divide the dataset into a training set (80%) and a test set (20%).
3. **Feature Scaling**: Standardizing features by removing the mean and scaling to unit variance.
4. **Training Models**: Training various classification models including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Kernel SVM, Naive Bayes, Decision Tree Classification, and Random Forest Classification using the sklearn library.
5. **Model Evaluation**: Using the test set data to predict loan approval and evaluating model performance with the confusion matrix and accuracy score.


## Conclusion
After training and evaluating all models, the performance was compared using the accuracy classification score. The results are summarized below:
- **Logistic Regression**: Accuracy Score = 0.78125
- **K-Nearest Neighbors (KNN)**: Accuracy Score = 0.69792
- **Support Vector Machine (SVM)**: Accuracy Score = 0.82292
- **Kernel SVM**: Accuracy Score = 0.82292
- **Naive Bayes**: Accuracy Score = 0.69792
- **Decision Tree Classification**: Accuracy Score = 0.72917
- **Random Forest Classification**: Accuracy Score = 0.77083

The analysis concludes that Kernel SVM demonstrates the highest performance among the evaluated models for predicting loan approval based on the dataset used.

