import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime
import os

# Loading the dataset
df = pd.read_csv('datasets\diabetesDataset.csv', usecols=['Glucose', 'Insulin', 'BMI', 'Age', 'Outcome'])

# Model Building for SVM
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating SVM Model
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

svm_accuracy = svm_classifier.score(X_test, y_test)
svm_accuracy = f"{svm_accuracy:.3f}"

# Save the trained SVM model as a pickle file
svm_model_filename = 'diabetes-prediction-svm-model.pkl'
with open(svm_model_filename, 'wb') as svm_model_file:
    pickle.dump(svm_classifier, svm_model_file)

print(f'SVM Model saved as {svm_model_filename}')

# Model Building for Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Logistic Regression Model
logreg_classifier = LogisticRegression(random_state=42)
logreg_classifier.fit(X_train, y_train)

# Evaluate the Logistic Regression model
logreg_accuracy = logreg_classifier.score(X_test, y_test)
logreg_accuracy = f"{logreg_accuracy:.3f}"

# Save the trained Logistic Regression model as a pickle file
logreg_model_filename = 'diabetes-prediction-logistic-regression-model.pkl'
with open(logreg_model_filename, 'wb') as logreg_model_file:
    pickle.dump(logreg_classifier, logreg_model_file)

print(f'Logistic Regression Model saved as {logreg_model_filename}')

# Store model accuracies and models with time and date in a DataFrame
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data = {
    'Model': ['SVM', 'Logistic Regression'],
    'Accuracy': [svm_accuracy, logreg_accuracy],
    'Model_Filename': [svm_model_filename, logreg_model_filename],
    'Timestamp': [current_time, current_time]
}

model_details_df = pd.DataFrame(data)

model_details_filename = 'datasets\model_details.csv'

# Check if the file exists
if os.path.exists(model_details_filename):
    # Load existing file and append new data
    existing_data = pd.read_csv(model_details_filename)
    updated_data = pd.concat([existing_data, model_details_df], ignore_index=True)
    updated_data.to_csv(model_details_filename, index=False)
else:
    # Save model details to a new file
    model_details_df.to_csv(model_details_filename, index=False)

print(f"Models' details saved to {model_details_filename}")
