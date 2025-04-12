import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imb_Pipeline
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("data/Customer_Data.csv")
df = df[df['Customer_Status'] != 'Joined']
df['Churn'] = df['Customer_Status'].apply(lambda x: 1 if x == 'Churned' else 0)
drop_cols = ['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason', 'Value_Deal',
             'Streaming_TV', 'Streaming_Movies', 'Streaming_Music']
df.drop(columns=drop_cols, inplace=True)

# Identify columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Churn')

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Split data
X = df.drop(columns='Churn')
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline with SMOTE
pipeline = imb_Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy='minority')),
    ('classifier', GradientBoostingClassifier(
        random_state=42,
        n_iter_no_change=5,
        validation_fraction=0.1
    ))
])

# Parameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 4],
    'classifier__subsample': [0.9, 1.0],
    'classifier__min_samples_split': [5, 10]
}

# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTest Performance:")
print(classification_report(y_test, y_pred))
print(f"Test ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Test Average Precision: {average_precision_score(y_test, y_prob):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/churn_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")