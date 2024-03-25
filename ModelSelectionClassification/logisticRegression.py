import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Importing the dataset
dataset = pd.read_csv('loan_data.csv')
# Checking for missing values
print(dataset.isnull().sum())
print(dataset.duplicated().sum())

# Handle missing values
# Manually specify columns to impute
numerical_columns_to_impute = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term']
categorical_columns_to_impute = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

# Imputers
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Impute specified numerical columns and categorical columns
dataset[numerical_columns_to_impute] = num_imputer.fit_transform(dataset[numerical_columns_to_impute])
dataset[categorical_columns_to_impute] = cat_imputer.fit_transform(dataset[categorical_columns_to_impute])

dataset['Dependents'] = dataset['Dependents'].replace('3+', 3).astype(int) # Replace '3+' with 3 and convert to int

# Checking for missing values
print(dataset.isnull().sum())
print(dataset.duplicated().sum())
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [0, 1, 2, 3, 4, 10]),  # One-hot encoder for categorical columns
    ],
    remainder='passthrough'
)

X_transformed = ct.fit_transform(X)
y = LabelEncoder().fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))




