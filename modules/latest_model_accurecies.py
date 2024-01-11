import pandas as pd

def get_latest_model_accuracies():
    # Read data from the model_details.csv file
    model_details_filename = 'model_details.csv'
    try:
        # Load the CSV file using pandas
        df = pd.read_csv(model_details_filename)

        # Filter data for SVM and Logistic Regression models
        svm_data = df[df['Model'] == 'SVM']
        logreg_data = df[df['Model'] == 'Logistic Regression']

        # Get the latest accuracies for SVM and Logistic Regression models
        latest_svm_accuracy = svm_data['Accuracy'].iloc[-1] if not svm_data.empty else 'N/A'
        latest_logreg_accuracy = logreg_data['Accuracy'].iloc[-1] if not logreg_data.empty else 'N/A'

    except FileNotFoundError:
        # Handle if the file doesn't exist or any other exception
        latest_svm_accuracy = 'N/A'
        latest_logreg_accuracy = 'N/A'

    return latest_svm_accuracy, latest_logreg_accuracy