# ChurnAI — Customer Churn Prediction System

**An end-to-end Machine Learning web application for predicting customer churn using Flask and Random Forest.**

<p align="center">
  <strong>Live Demo:</strong>
  <a href="https://churn-prediction-flask.onrender.com">ChurnAI Web Application</a>
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Input Features](#input-features)
- [Project Structure](#project-structure)
- [How the Application Works](#how-the-application-works)
- [Installation and Setup](#installation-and-setup)
- [Run the Application Locally](#run-the-application-locally)
- [Model Training](#model-training)
- [Deployment on Render](#deployment-on-render)
- [Database](#database)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Limitations](#limitations)
- [Author](#author)
- [License](#license)

---

## Project Overview

**ChurnAI** is a Machine Learning-based web application designed to predict whether a customer is likely to leave a company or discontinue its services.

Customer churn is a major challenge for businesses that depend on customer retention, including subscription services, telecommunications, banking, e-commerce, and other customer-oriented industries.

This application uses a trained **Random Forest classification model** to analyze customer information and predict churn. It provides a web-based interface where users can enter customer details, view prediction results, and access a dashboard for prediction history.

The application is developed using Python and Flask, with a trained Machine Learning model, a web interface, and an SQLite database.

**Live Application:**  
https://churn-prediction-flask.onrender.com

**GitHub Repository:**  
https://github.com/Om-Pawar-9834/churn-prediction-flask

## Problem Statement

Businesses often lose customers because of dissatisfaction, poor service experiences, payment delays, low engagement, or other factors.

Identifying customers who may leave is important because it allows businesses to take preventive actions, improve customer satisfaction, and develop better retention strategies.

Manual analysis of customer information can be time-consuming and difficult when dealing with large datasets.

ChurnAI addresses this problem by using Machine Learning to analyze customer attributes and generate a churn prediction through a simple web application.

## Objectives

The primary objectives of this project are:

1. Develop a Machine Learning model to classify customers according to churn risk.
2. Build a user-friendly web interface using Flask and frontend technologies.
3. Process customer information and generate predictions using a trained model.
4. Display prediction results in an understandable format.
5. Maintain prediction history using an SQLite database.
6. Provide a dashboard to review recorded predictions.
7. Deploy the application online so that users can access it through a web browser.
8. Demonstrate the integration of Machine Learning, backend development, database management, and cloud deployment.

## Key Features

### 1. Customer Churn Prediction

Users can enter customer information through an interactive form. The application processes the submitted values and sends them to the trained Machine Learning model.

The model predicts whether the customer is classified as:

- **Churn:** The model predicts that the customer belongs to the churn class.
- **Not Churn:** The model predicts that the customer belongs to the non-churn class.

The result depends on the model's training data and learned patterns.

### 2. Customer Information Form

The prediction form accepts the following information:

- Age
- Gender
- Customer tenure
- Usage frequency
- Number of support calls
- Payment delay
- Subscription type
- Contract length
- Total customer spending
- Days since the last interaction

### 3. Prediction Result Interface

The application displays the prediction result after the customer information is submitted.

The result interface helps users understand the model's classification without needing to interact directly with Python code.

### 4. Dashboard

The dashboard provides access to recorded prediction history and helps users review previous prediction requests.

The exact statistics and records displayed depend on the application's implementation and the data stored in its database.

### 5. About Model Page

The application includes an About Model page that provides information about the Machine Learning model and the prediction system.

### 6. Database Integration

SQLite is used to store prediction records locally through the application's database layer.

This allows the application to retrieve previously recorded predictions for the dashboard.

### 7. Online Deployment

The application is deployed on Render and can be accessed through an HTTPS URL.

The source code is maintained on GitHub, allowing changes to be version-controlled and deployed through the connected repository.

### 8. Responsive Web Interface

The application uses web frontend technologies to present customer information forms, navigation, prediction results, and dashboard content.

The interface is designed for convenient access through a web browser.

---

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Machine Learning Algorithm | Random Forest Classifier |
| Data Processing | pandas, NumPy |
| Machine Learning Library | scikit-learn |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Model Serialization | Python pickle |
| Production Web Server | Gunicorn |
| Version Control | Git |
| Source Code Hosting | GitHub |
| Cloud Deployment | Render |
| Deployment Protocol | HTTPS |

---

## Machine Learning Workflow

The project follows a typical supervised Machine Learning workflow.

### Step 1: Dataset

The model is trained using customer churn data containing customer attributes and a target variable representing churn.

The dataset provides examples from which the model learns relationships between customer information and churn outcomes.

### Step 2: Data Preprocessing

Before training, the dataset must be prepared for Machine Learning.

Typical preprocessing operations include:

- Inspecting the dataset.
- Checking data types and missing values.
- Identifying the target variable.
- Separating input features from the target.
- Encoding categorical variables where required.
- Preparing the data in the format expected by the model.

The exact preprocessing steps depend on the training code.

### Step 3: Feature Selection

The model uses customer-related features such as tenure, usage frequency, support calls, payment delay, subscription type, contract length, and spending.

These features provide information that may help distinguish customers who churn from customers who do not.

### Step 4: Model Training

The project uses a Random Forest classification model.

Random Forest combines predictions from multiple decision trees to produce a final classification. It can learn nonlinear relationships and interactions among customer features.

### Step 5: Model Serialization

The trained model or preprocessing pipeline is saved as a serialized file:

`model/model.pkl`

The application loads this file when it needs to make predictions.

The saved artifact must remain compatible with the Python and scikit-learn environment used to run the application.

### Step 6: Prediction

When a user submits the customer form:

1. Flask receives the form data.
2. The application validates and prepares the input.
3. The values are arranged in the feature order expected by the trained model.
4. The model generates a prediction.
5. The result is returned to the web interface.
6. The application records the prediction when database recording is enabled.

### Step 7: Evaluation

A complete Machine Learning evaluation should use a held-out test dataset and appropriate classification metrics, such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC, when applicable

These metrics help assess how well the model identifies churn cases. Actual model performance should be reported using measured evaluation results rather than assumed values.

---

## Input Features

The following table describes the customer attributes accepted by the prediction interface.

| Feature | Description |
|---|---|
| Age | Age of the customer |
| Gender | Customer's gender category |
| Tenure | Duration of the customer relationship, in months |
| Usage Frequency | Number of usage instances per month |
| Support Calls | Number of customer support calls |
| Payment Delay | Payment delay in days |
| Subscription Type | Customer's subscription category |
| Contract Length | Duration category of the customer's contract |
| Total Spend | Total spending recorded for the customer |
| Last Interaction | Number of days since the customer's last interaction |

**Important:** Input values must match the data types, category labels, units, and feature order expected by the trained model. The form's labels alone do not determine the model's internal feature schema.

---

## Project Structure

The repository is organized into the following main components:

```text
churn-prediction-flask/
│
├── model/
│   ├── model.pkl
│   ├── train_model.py
│   └── customer_churn_dataset-testing-master.csv
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── about.html
│
├── app.py
├── predictions.db
├── requirements.txt
├── Procfile
├── .gitignore
├── .python-version
├── run_instructions.txt
└── README.md
```

*Note: The names of individual template and static files may differ in the actual repository. Keep the README structure synchronized with the files committed to GitHub.*

### Important Files

| File or Directory | Purpose |
|---|---|
| `app.py` | Flask application, routes, prediction handling, and database operations |
| `model/model.pkl` | Saved trained model or preprocessing pipeline |
| `model/train_model.py` | Script used to train and save the model |
| `model/` | Model artifacts and available training data |
| `templates/` | HTML templates for the application's pages |
| `static/` | Frontend assets such as CSS, JavaScript, and images |
| `predictions.db` | SQLite database file included in the repository |
| `requirements.txt` | Python package dependencies |
| `Procfile` | Defines the production start command |
| `.python-version` | Specifies the Python version for deployment environments that support this file |
| `run_instructions.txt` | Additional project execution instructions |
| `README.md` | Project documentation |

---

## Installation and Setup

Follow these steps to run the project on your local machine.

### Prerequisites

Install the following software:

- Python 3.11.9
- Git
- Visual Studio Code or another Python-compatible editor
- pip, Python's package installer

Python 3.11.9 is the intended project version. Use a compatible environment for the pinned dependencies.

### Step 1: Clone the Repository

Open a terminal and execute:

```bash
git clone https://github.com/Om-Pawar-9834/churn-prediction-flask.git
```

### Step 2: Navigate to the Project Directory

```bash
cd churn-prediction-flask
```

### Step 3: Create a Virtual Environment

On Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, you can use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify the Model File

Confirm that the following file exists:

```text
model/model.pkl
```

The application needs the saved model artifact to make predictions. If it is missing, train the model using the project's training script and verify that the resulting artifact is saved at the expected path.

---

## Run the Application Locally

After installing dependencies, start the Flask application.

```bash
python app.py
```

Open the local address configured by the application. If Flask is using its default development port, visit:

http://127.0.0.1:5000/

You should be able to access the prediction interface, enter customer information, and view the result.

**Note:** The development server is intended for local development. Use the configured production server command when deploying the application.

---

## Model Training

The repository includes a training script named `model/train_model.py`.

If you need to retrain the model, first inspect the script to confirm the expected dataset path, target column, preprocessing, feature order, and output location.

Run the script from the project root:

```bash
python model/train_model.py
```

If the script expects a different working directory or dataset path, follow the paths specified in the script.

After training:

1. Confirm that training completes without errors.
2. Verify the saved model file exists.
3. Ensure the saved model uses the expected input features and preprocessing.
4. Test predictions locally.
5. Deploy the updated model with the application.

**Important:** Retraining the model may change its predictions. Evaluate the new model on suitable test data before replacing a previously working production artifact.

---

## Deployment on Render

The application is deployed on Render as a Python web service.

### Live Application

https://churn-prediction-flask.onrender.com

### Deployment Configuration

| Setting | Value |
|---|---|
| Service Type | Web Service |
| Runtime | Python |
| Branch | `main` |
| Root Directory | Repository root |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn --bind 0.0.0.0:$PORT app:app` |
| Python Version | `3.11.9` |
| Source Repository | GitHub |

The Python version should be set using the configuration supported by the Render service. The `.python-version` file in the repository contains:

```text
3.11.9
```

If the Render service has an overriding Python version setting, verify that it is also configured appropriately.

### Deployment Steps

1. Push the project source code to GitHub.
2. Sign in to Render.
3. Create a new Web Service or select the existing service.
4. Connect the GitHub repository.
5. Select the `main` branch.
6. Configure the build and start commands.
7. Ensure the Python runtime and package dependencies are compatible.
8. Deploy the service.
9. Review build logs for dependency or startup errors.
10. Open the generated HTTPS URL and test the application.

### Updating the Application

When automatic deployment is enabled, pushing a new commit to the configured branch can trigger a new deployment.

Typical Git commands:

```bash
git add .
git commit -m "Describe your changes"
git push origin main
```

After pushing, verify the deployment status and inspect the logs if the deployment fails.

### Production Considerations

- The application must bind to the port provided by Render through `$PORT`.
- The saved model file must be included in the deployment or downloaded through a configured model-artifact process.
- The installed scikit-learn version must be compatible with the serialized model.
- SQLite persistence on a free hosting instance should not be assumed to be permanent across restarts, redeployments, or instance replacement. Use persistent storage or a managed database if prediction history must be retained.
- Never commit API keys, passwords, private credentials, or other secrets to GitHub.

---

## Database

ChurnAI uses SQLite for prediction records.

SQLite is a lightweight, file-based relational database that does not require a separate database server.

The database supports storing and retrieving prediction history for the dashboard, according to the application's implementation.

### Database Considerations

- The application must have permission to read and write the database file.
- Database schema changes should be handled carefully.
- If the database file is missing, the application must initialize the required tables where appropriate.
- The database should not be treated as permanent cloud storage on an ephemeral deployment filesystem.
- For a production deployment that requires durable prediction history, consider a managed PostgreSQL database or another persistent database service.

---

## Requirements

The project's `requirements.txt` contains the following dependencies:

```text
Flask==3.1.2
gunicorn==23.0.0
pandas==1.5.2
numpy==1.26.4
scikit-learn==1.9.1
```

### Dependency Purposes

| Package | Purpose |
|---|---|
| Flask | Handles HTTP requests, routes, templates, and web application logic |
| Gunicorn | Runs the Flask application using a production WSGI server |
| pandas | Provides tools for loading and processing tabular data |
| NumPy | Supports numerical operations and array processing |
| scikit-learn | Provides the Machine Learning model and preprocessing tools |

The dependency versions should be tested together in the target Python environment. In particular, the version of scikit-learn used to load a serialized model should match the version used when the model was saved, or be verified as compatible.

---

## Troubleshooting

### 1. Pandas Installation Fails

**Possible cause:** Python and the pinned pandas version are incompatible, or a suitable prebuilt wheel is unavailable.

**Solution:**

- Use the intended Python version, `3.11.9`.
- Confirm that Render recognizes the version configuration.
- Reinstall dependencies using the project's requirements file.
- Review the complete build log for the first actual error.

### 2. `ModuleNotFoundError`

**Possible cause:** A required package is not installed, or the application imports a module that is missing from the deployment environment.

**Solution:**

- Check `requirements.txt`.
- Install the required dependency.
- Confirm that the correct virtual environment is active locally.
- Redeploy and inspect the build logs.

### 3. Model File Not Found

**Possible cause:** The model file is missing, its path is incorrect, or the working directory differs from what the application expects.

**Solution:**

- Confirm that `model/model.pkl` exists.
- Check the model-loading path in `app.py`.
- Ensure the artifact is available in the deployed repository or storage location.

### 4. Model Loading Error

**Possible cause:** The serialized model was saved using an incompatible Python or scikit-learn environment.

**Solution:**

- Check the version used to train and serialize the model.
- Match the deployment environment to the training environment where practical.
- Retrain and serialize the model in the intended environment if necessary.

### 5. Application Does Not Start

**Possible cause:** An incorrect start command, an application import error, or a missing dependency.

**Solution:**

Verify the start command:

```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

Also confirm that `app.py` exposes a Flask application object named `app`.

### 6. Prediction Fails

**Possible cause:** Invalid inputs, incorrect feature order, unexpected categorical values, or a mismatch between the form and the trained model.

**Solution:**

- Check the validation logic.
- Confirm the input feature names and order.
- Verify the preprocessing steps used during training.
- Test the prediction locally and inspect the server logs.

### 7. Dashboard History Is Missing

**Possible cause:** The database file was reset, the deployed filesystem is ephemeral, or the application is connecting to a different database path.

**Solution:**

- Check the database path and initialization logic.
- Inspect the deployed environment's storage configuration.
- Use persistent storage or a managed database when records must survive redeployments.

### 8. First Request Is Slow

Render free web services can spin down after inactivity and may take time to start when a new request arrives.

This initial delay does not necessarily mean the Machine Learning application is broken.

---

## Future Enhancements

Possible future improvements include:

- Display model confidence or calibrated churn probability, if properly supported by the trained model.
- Add interactive charts for prediction history and customer trends.
- Improve data validation and error handling.
- Add user authentication and role-based access.
- Introduce batch prediction for CSV uploads.
- Add explainability using feature importance or SHAP.
- Evaluate multiple classification algorithms and compare their performance.
- Add cross-validation and hyperparameter tuning.
- Introduce automated tests for routes, preprocessing, and predictions.
- Add structured application logging and deployment health checks.
- Migrate prediction history to a persistent managed database.
- Build a monitoring process to detect changes in model performance.
- Improve responsive design and accessibility.

These enhancements are potential extensions and are not necessarily implemented in the current version.

---

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- Model predictions are estimates and should not be treated as guaranteed outcomes.
- Performance may vary across customer populations and business environments.
- The application depends on the availability and compatibility of the saved model artifact.
- SQLite data may not persist reliably across hosting restarts without persistent storage.
- The application does not automatically guarantee a business retention strategy; predictions should be interpreted alongside business context.
- Model performance should be measured and documented using actual evaluation results.

---

## Author

**Om Pawar**

B.Tech — Artificial Intelligence and Data Science

GitHub: [Om-Pawar-9834](https://github.com/Om-Pawar-9834)

LinkedIn: [Om Pawar](https://www.linkedin.com/in/om-pawar-a5307b330/)

---

## License

No license has been specified for this repository yet.

If you intend to make this project reusable by others, consider adding a suitable open-source license, such as the MIT License, after deciding how you want others to use and distribute your work.

---

## Acknowledgements

This project demonstrates the practical integration of Python, Machine Learning, Flask web development, database operations, GitHub version control, and cloud deployment.

Thank you for exploring **ChurnAI — Customer Churn Prediction System**.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
