# Importing essential libraries
from flask import flash, Flask, render_template, request, redirect, url_for, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from user import User  
import pickle
import requests
import pandas as pd
import numpy as np
import csv
import subprocess
import shutil
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from modules.chat import get_response


from modules.calculate_bmi import calculate_bmi
from modules.save_users_data_in_csv_file import save_to_csv
#from modules.calculate_output_accuracy import calculate_ans_accuracy
from modules.latest_model_accurecies import get_latest_model_accuracies


app = Flask(__name__)
app.config['SECRET_KEY'] = 'd4d5e6f7g8h9j0k'

# Set up Flask-Login
login_manager = LoginManager(app)

agreement_confirmed = False

# Example: a dictionary to store user information
users = {
    'Nimantha': {'password': os.getenv('USER_NIMANTHA_PASSWORD')},
    'test': {'password': os.getenv('USER_TEST_PASSWORD')}
}

# Load the SVM Classifier model
svm_filename = 'diabetes-prediction-svm-model.pkl'
svm_classifier = pickle.load(open(svm_filename, 'rb'))

# Load the Logistic Regression Classifier model
logreg_filename = 'diabetes-prediction-logistic-regression-model.pkl'
logreg_classifier = pickle.load(open(logreg_filename, 'rb'))

# File to store user data
csv_filename = os.path.join('datasets', 'user_data.csv')

try:
    with open(csv_filename, 'r', newline='') as file:
        pass
except FileNotFoundError:
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Weight', 'Height', 'BMI', 'Age', 'Prediction'])

@login_manager.user_loader
def load_user(user_id):
    # This function is used by Flask-Login to get a user object based on the user ID stored in the session
    user_info = users.get(user_id)
    if user_info:
        return User(user_id, user_info['password'])
    return None

@app.route('/')
def home():
    global agreement_confirmed
    if agreement_confirmed:
        # If agreement is confirmed, render the dashboard
        return render_template('index.html', agreement_confirmed=True)
    else:
        # If agreement is not confirmed, render the confirmation form
        return render_template('index.html', agreement_confirmed=False)

# Route to handle confirmation form submission
@app.route('/confirm', methods=['POST'])
def confirm():
    global agreement_confirmed
    if request.method == 'POST':
        agreement = request.form.get('agreement')
        if agreement == 'on':  # The checkbox is checked
            agreement_confirmed = True
            # Redirect to the home page (dashboard)
            return redirect(url_for('home'))
    # If agreement is not confirmed or method is not POST, stay on the home page
    return redirect(url_for('home'))

# Dashboard route (index.html)
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')  # Render the dashboard template

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        glucose = float(request.form['glucose'])
        insulin = float(request.form['insulin'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = int(request.form['age'])
        
        # Calculate BMI
        bmi = calculate_bmi(weight, height)

        latest_svm_accuracy, latest_logreg_accuracy,model_of_use = get_latest_model_accuracies()

        if latest_svm_accuracy >= latest_logreg_accuracy:
            # Prediction using SVM model
            svm_data = np.array([[glucose, insulin, bmi, age]])
            svm_prediction = svm_classifier.predict(svm_data)
            prediction = svm_prediction[0]
            model_used = 'SVM'
        else:
        # Prediction using Logistic Regression model
            logreg_data = np.array([[glucose, insulin, bmi, age]])
            logreg_prediction = logreg_classifier.predict(logreg_data)
            prediction = logreg_prediction[0]
            model_used = 'Logistic Regression'


        # Save data, prediction, accuracy, and model used to CSV file
        save_to_csv([glucose, insulin, bmi, age, prediction])

        return render_template('result.html', prediction=prediction, model=model_used,
                               weight=weight, height=height, age=age,bmi=bmi,glucose=glucose,insulin=insulin)

@app.route('/contactUs', methods=['GET', 'POST'])
def contactUs():
    return render_template('contactUs.html')

@app.route('/submit_contact_form', methods=['POST'])
def submit_contact_form():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Compose email
        subject = 'Contact Form Submission'
        body = f'Name: {name}\nEmail: {email}\nMessage: {message}'

        # Send email
        send_email(subject, body)

        return 'Thank you for your message. We will get back to you soon.'

    return 'Failed to submit contact form. Please try again.'

def send_email(subject, body):
    # Add your email configuration
    EMAIL_ADDRESS = 'sumekasu2320@gmail.com'
    EMAIL_ADDRESS2 = 'nimanthakasun2000@gmail.com'
    EMAIL_PASSWORD = 'nn07235226d12nn'

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS2
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

# Admin login route
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        recaptcha_response = request.form.get('g-recaptcha-response')
        secret_key = '6Lc-vmwpAAAAAGWkZH1U0H3MamR52V3Q9rP4zg6g'

        # Verify reCAPTCHA response
        response = requests.post('https://www.google.com/recaptcha/api/siteverify', 
                                data={'secret': secret_key, 'response': recaptcha_response})

        data = response.json()
        if data['success']:
            user_info = users.get(username)
            if user_info and user_info['password'] == password:
                login_user(User(username, user_info['password']))
                return redirect(url_for('admin_dashboard'))
        else:
            # reCAPTCHA verification failed
            return render_template('admin_login.html')
        
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    # Read data from the user_data.csv file
    with open('datasets/user_data.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Limit the data to the first 10 rows
    limited_data = data[:]
    
    latest_svm_accuracy, latest_logreg_accuracy,model_of_use = get_latest_model_accuracies()

    return render_template('admin_dashboard.html', data=limited_data, accuracy_svm=latest_svm_accuracy, accuracy_logreg=latest_logreg_accuracy, username=current_user.get_id(), model_of_use=model_of_use)

# Logout route
@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    return redirect(url_for('home'))

# Run DiabetesPredictorDeployment.py route
@app.route('/run_deployment')
@login_required
def run_deployment():
    # Replace 'DiabetesPredictorDeployment.py' with the actual filename if it's different
    deployment_script = 'DiabetesPredictorDeployment.py'
    
    # Run the deployment script using subprocess
    subprocess.run(['python', deployment_script])
    
    return render_template('deployment_success.html')  # Create a success page or redirect as needed

# Backup route
@app.route('/backup_files')
@login_required
def backup_files():
    # Specify the path to the backup folder
    backup_folder = 'backup/'

    try:
        # Create the backup folder if it doesn't exist
        os.makedirs(backup_folder, exist_ok=True)

        # Copy current .pkl files to the backup folder
        shutil.copy('diabetes-prediction-svm-model.pkl', backup_folder)
        shutil.copy('diabetes-prediction-logistic-regression-model.pkl', backup_folder)

        flash("Backup successful. .pkl files are copied to the backup folder.", 'success')
    except Exception as e:
        flash(f"Backup failed. Error: {str(e)}", 'danger')

    return redirect(url_for('admin_dashboard'))

# Route to append user_data.csv to diabetesDataset.csv for train
@app.route('/append_data')
@login_required
def append_data():
    try:
        # Read data from user_data.csv (excluding the header row)
        with open('datasets/user_data.csv', 'r') as user_file:
            user_reader = csv.reader(user_file)
            user_data = list(user_reader)[1:]

        # Append data to diabetesDataset.csv
        with open('datasets\diabetesDataset.csv', 'a', newline='') as dataset_file:
            dataset_writer = csv.writer(dataset_file)
            dataset_writer.writerows(user_data)

        flash("Data appended successfully to diabetesDataset.csv.", 'success')
    except Exception as e:
        flash(f"Append data failed. Error: {str(e)}", 'danger')

    return redirect(url_for('admin_dashboard'))

# Route to delete a specific row
@app.route('/delete_row/<int:index>')
@login_required
def delete_row(index):
    try:
        # Read data from user_data.csv
        with open('datasets/user_data.csv', 'r') as user_file:
            user_reader = csv.reader(user_file)
            user_data = list(user_reader)

        # Delete the specified row
        del user_data[index+1]

        # Write the modified data back to user_data.csv
        with open('datasets/user_data.csv', 'w', newline='') as user_file:
            user_writer = csv.writer(user_file)
            user_writer.writerows(user_data)

        flash("Row deleted successfully.", 'success')
    except Exception as e:
        flash(f"Delete row failed. Error: {str(e)}", 'danger')

    return redirect(url_for('admin_dashboard'))

@app.get("/")
def index_get():
    return render_template("base.html")

# Configure rate limiting
limiter = Limiter(
    app,
    default_limits=["5 per minute"],  # Adjust the rate limit as needed
)

@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute")  # Adjust the rate limit as needed
def predict():
    try:
        data = request.json
        message = data.get("message")

        if not message:
            return jsonify({"error": "Invalid input"}), 400

        response = get_response(message)

        if response is None:
            app.logger.error("No response found for input: %s", message)
            return jsonify({"error": "Internal Server Error"}), 500

        return jsonify({"answer": response})

    except Exception as e:
        app.logger.exception("An error occurred during request processing: %s", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
