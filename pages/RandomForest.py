import subprocess
import sys
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

pip install scikit-learn


# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, random_state=0) # You can adjust hyperparameters
#rf_classifier = RandomForestClassifier(n_estimators=100,random_state=0) # You can adjust hyperparameters
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
st.write(classification_report(y_test, y_pred))
st.write(confusion_matrix(y_test, y_pred))


# Save the trained model to a pickle file
#filename = 'random_forest_model.pkl'pickle.dump(rf_classifier, open(filename, 'wb'))
with open("random_forest_model.pkl", "wb") as file:
  pickle.dump(rf_classifier, file)
