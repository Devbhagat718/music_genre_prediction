# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load and preprocess data
dataframe_for_3sec = pd.read_csv('features_3_sec.csv')
X = dataframe_for_3sec.drop(columns=['label', 'filename'])
y = dataframe_for_3sec['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling, polynomial features, and PCA
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial Features and PCA for dimensionality reduction
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Define models and Voting Classifier
log_reg = LogisticRegression(max_iter=1000, random_state=42)
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
ensemble_model = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rf_model), ('svm', svm_model)], voting='hard'
)

# Train model
ensemble_model.fit(X_train_poly, y_train)

# Save models and scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')
joblib.dump(ensemble_model, 'model.pkl')

print("Model and scalers saved successfully!")
