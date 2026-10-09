# ============================================================
# CUSTOMER CHURN PREDICTION
# COMPLETE MODEL TRAINING FILE
# ============================================================

from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# STEP 1: PATHS
# ============================================================

MODEL_FOLDER = Path(__file__).resolve().parent

DATASET_PATH = (
    MODEL_FOLDER /
    "customer_churn_dataset-testing-master.csv"
)

MODEL_PATH = (
    MODEL_FOLDER /
    "model.pkl"
)


print("\n")
print("=" * 70)
print("CUSTOMER CHURN MODEL TRAINING")
print("=" * 70)

print("\nModel folder:")
print(MODEL_FOLDER)

print("\nDataset path:")
print(DATASET_PATH)

print("\nModel will be saved at:")
print(MODEL_PATH)


# ============================================================
# STEP 2: CHECK DATASET
# ============================================================

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"""
Dataset not found!

Expected location:
{DATASET_PATH}

Make sure the CSV file is inside the same folder as
train_model.py.
"""
    )


# ============================================================
# STEP 3: LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)


print("\n")
print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 4: BASIC INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

df.info()


# ============================================================
# STEP 5: MISSING VALUES
# ============================================================

print("\n")
print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# ============================================================
# STEP 6: DUPLICATE VALUES
# ============================================================

print("\n")
print("=" * 70)
print("DUPLICATES")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print("Duplicate rows removed.")

else:

    print("No duplicate rows found.")


# ============================================================
# STEP 7: REMOVE CUSTOMER ID
# ============================================================

if "CustomerID" in df.columns:

    df.drop(
        "CustomerID",
        axis=1,
        inplace=True
    )

    print("\nCustomerID removed.")


# ============================================================
# STEP 8: TARGET COLUMN
# ============================================================

TARGET_COLUMN = "Churn"


if TARGET_COLUMN not in df.columns:

    raise ValueError(
        """
ERROR:

'Churn' column was not found in the dataset.
"""
    )


print("\n")
print("=" * 70)
print("TARGET COLUMN")
print("=" * 70)

print("Target:", TARGET_COLUMN)

print("\nTarget values:")
print(df[TARGET_COLUMN].unique())

print("\nTarget distribution:")
print(df[TARGET_COLUMN].value_counts())


# ============================================================
# STEP 9: DEFINE EXACT FEATURES
# ============================================================

FEATURE_COLUMNS = [

    "Age",

    "Gender",

    "Tenure",

    "Usage Frequency",

    "Support Calls",

    "Payment Delay",

    "Subscription Type",

    "Contract Length",

    "Total Spend",

    "Last Interaction"

]


print("\n")
print("=" * 70)
print("FEATURES USED FOR TRAINING")
print("=" * 70)

for feature in FEATURE_COLUMNS:

    print(" -", feature)


# ============================================================
# STEP 10: CHECK FEATURES
# ============================================================

missing_features = [

    column
    for column in FEATURE_COLUMNS
    if column not in df.columns

]


if missing_features:

    raise ValueError(

        "\nThe following required features "
        "are missing from the dataset:\n"
        + str(missing_features)

    )


# ============================================================
# STEP 11: CREATE X AND y
# ============================================================

X = df[FEATURE_COLUMNS].copy()

y = df[TARGET_COLUMN].copy()


print("\n")
print("=" * 70)
print("X AND y CREATED")
print("=" * 70)

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# ============================================================
# STEP 12: TARGET CONVERSION
# ============================================================

print("\n")
print("=" * 70)
print("TARGET PROCESSING")
print("=" * 70)


# If target is already numeric, keep it.
#
# If target is Yes/No, convert it to 1/0.

if y.dtype == "object":

    y = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "churn": 1,
            "no churn": 0
        })
    )


else:

    y = pd.to_numeric(
        y,
        errors="coerce"
    )


# Remove rows where target could not be converted.

valid_rows = y.notnull()

X = X.loc[valid_rows].copy()

y = y.loc[valid_rows].copy()


# Convert target to integer.

y = y.astype(int)


print("\nFinal target values:")
print(y.value_counts())


# ============================================================
# STEP 13: CONVERT NUMERICAL COLUMNS
# ============================================================

# IMPORTANT:
#
# Contract Length is NOT here.
#
# Contract Length contains values such as:
#
# Monthly
# Quarterly
# Annual
#
# Therefore it must be treated as categorical.


NUMERICAL_COLUMNS = [

    "Age",

    "Tenure",

    "Usage Frequency",

    "Support Calls",

    "Payment Delay",

    "Total Spend",

    "Last Interaction"

]


# ============================================================
# STEP 14: CATEGORICAL COLUMNS
# ============================================================

# IMPORTANT:
#
# Contract Length is included here.
#
# This fixes the error:
#
# Cannot use median strategy with non-numeric data:
# could not convert string to float: 'Monthly'


CATEGORICAL_COLUMNS = [

    "Gender",

    "Subscription Type",

    "Contract Length"

]


print("\n")
print("=" * 70)
print("NUMERICAL COLUMNS")
print("=" * 70)

print(NUMERICAL_COLUMNS)


print("\n")
print("=" * 70)
print("CATEGORICAL COLUMNS")
print("=" * 70)

print(CATEGORICAL_COLUMNS)


# ============================================================
# STEP 15: CONVERT NUMERICAL DATA
# ============================================================

for column in NUMERICAL_COLUMNS:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# STEP 16: BASIC EDA
# ============================================================

print("\n")
print("=" * 70)
print("STARTING EDA")
print("=" * 70)


# ------------------------------------------------------------
# 16.1 Churn Distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    x=y
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    [0, 1],
    ["No Churn", "Churn"]
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16.2 Contract Length vs Churn
# ------------------------------------------------------------

eda_df = X.copy()

eda_df["Churn"] = y.values


plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=eda_df,
    x="Contract Length",
    hue="Churn"
)

plt.title(
    "Contract Length vs Churn"
)

plt.xlabel(
    "Contract Length"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16.3 Subscription Type vs Churn
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=eda_df,
    x="Subscription Type",
    hue="Churn"
)

plt.title(
    "Subscription Type vs Churn"
)

plt.xlabel(
    "Subscription Type"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16.4 Gender vs Churn
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    data=eda_df,
    x="Gender",
    hue="Churn"
)

plt.title(
    "Gender vs Churn"
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16.5 Tenure Distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data=eda_df,
    x="Tenure",
    hue="Churn",
    kde=True
)

plt.title(
    "Tenure Distribution by Churn"
)

plt.xlabel(
    "Tenure"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16.6 Age Distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    data=eda_df,
    x="Age",
    hue="Churn",
    kde=True
)

plt.title(
    "Age Distribution by Churn"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16.7 Support Calls vs Churn
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=eda_df,
    x="Churn",
    y="Support Calls"
)

plt.title(
    "Support Calls vs Churn"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Support Calls"
)

plt.xticks(
    [0, 1],
    ["No Churn", "Churn"]
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16.8 Payment Delay vs Churn
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=eda_df,
    x="Churn",
    y="Payment Delay"
)

plt.title(
    "Payment Delay vs Churn"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Payment Delay"
)

plt.xticks(
    [0, 1],
    ["No Churn", "Churn"]
)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 17: TRAIN TEST SPLIT
# ============================================================

print("\n")
print("=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nX_train:", X_train.shape)

print("X_test :", X_test.shape)

print("y_train:", y_train.shape)

print("y_test :", y_test.shape)


# ============================================================
# STEP 18: NUMERICAL PIPELINE
# ============================================================

print("\n")
print("=" * 70)
print("CREATING NUMERICAL PIPELINE")
print("=" * 70)


numerical_pipeline = Pipeline([

    (
        "imputer",

        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",

        StandardScaler()
    )

])


# ============================================================
# STEP 19: CATEGORICAL PIPELINE
# ============================================================

print("\n")
print("=" * 70)
print("CREATING CATEGORICAL PIPELINE")
print("=" * 70)


categorical_pipeline = Pipeline([

    (
        "imputer",

        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",

        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

])


# ============================================================
# STEP 20: COLUMN TRANSFORMER
# ============================================================

print("\n")
print("=" * 70)
print("CREATING COLUMN TRANSFORMER")
print("=" * 70)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",

            numerical_pipeline,

            NUMERICAL_COLUMNS
        ),

        (
            "categorical",

            categorical_pipeline,

            CATEGORICAL_COLUMNS
        )

    ]

)


# ============================================================
# STEP 21: LOGISTIC REGRESSION PIPELINE
# ============================================================

logistic_pipeline = Pipeline([

    (
        "preprocessor",

        preprocessor
    ),

    (
        "model",

        LogisticRegression(
            max_iter=1000
        )
    )

])


# ============================================================
# STEP 22: RANDOM FOREST PIPELINE
# ============================================================

random_forest_pipeline = Pipeline([

    (
        "preprocessor",

        preprocessor
    ),

    (
        "model",

        RandomForestClassifier(

            random_state=42,

            n_jobs=-1

        )

    )

])


# ============================================================
# STEP 23: TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)


logistic_pipeline.fit(

    X_train,

    y_train

)


lr_prediction = logistic_pipeline.predict(
    X_test
)


lr_accuracy = accuracy_score(
    y_test,
    lr_prediction
)

lr_precision = precision_score(
    y_test,
    lr_prediction,
    zero_division=0
)

lr_recall = recall_score(
    y_test,
    lr_prediction,
    zero_division=0
)

lr_f1 = f1_score(
    y_test,
    lr_prediction,
    zero_division=0
)


print("\nLogistic Regression Results:")

print(
    "Accuracy :",
    round(lr_accuracy, 4)
)

print(
    "Precision:",
    round(lr_precision, 4)
)

print(
    "Recall   :",
    round(lr_recall, 4)
)

print(
    "F1 Score :",
    round(lr_f1, 4)
)


# ============================================================
# STEP 24: TRAIN RANDOM FOREST
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


random_forest_pipeline.fit(

    X_train,

    y_train

)


rf_prediction = random_forest_pipeline.predict(
    X_test
)


rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

rf_precision = precision_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_prediction,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_prediction,
    zero_division=0
)


print("\nRandom Forest Results:")

print(
    "Accuracy :",
    round(rf_accuracy, 4)
)

print(
    "Precision:",
    round(rf_precision, 4)
)

print(
    "Recall   :",
    round(rf_recall, 4)
)

print(
    "F1 Score :",
    round(rf_f1, 4)
)


# ============================================================
# STEP 25: MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Random Forest"

    ],

    "Accuracy": [

        lr_accuracy,

        rf_accuracy

    ],

    "Precision": [

        lr_precision,

        rf_precision

    ],

    "Recall": [

        lr_recall,

        rf_recall

    ],

    "F1 Score": [

        lr_f1,

        rf_f1

    ]

})


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison)


# ============================================================
# STEP 26: SELECT MODEL FOR TUNING
# ============================================================

if rf_f1 >= lr_f1:

    selected_model = "Random Forest"

    tuning_pipeline = random_forest_pipeline

else:

    selected_model = "Logistic Regression"

    tuning_pipeline = logistic_pipeline


print("\n")
print("=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(selected_model)


# ============================================================
# STEP 27: HYPERPARAMETER GRID
# ============================================================

if selected_model == "Random Forest":

    param_grid = {

        "model__n_estimators": [

            100,

            200

        ],

        "model__max_depth": [

            None,

            10,

            20

        ],

        "model__min_samples_split": [

            2,

            5

        ],

        "model__min_samples_leaf": [

            1,

            2

        ]

    }


else:

    param_grid = {

        "model__C": [

            0.01,

            0.1,

            1,

            10,

            100

        ],

        "model__solver": [

            "liblinear",

            "lbfgs"

        ]

    }


# ============================================================
# STEP 28: GRID SEARCH CV
# ============================================================

print("\n")
print("=" * 70)
print("STARTING GRID SEARCH")
print("=" * 70)

print(
    "\nThis may take some time..."
)


grid_search = GridSearchCV(

    estimator=tuning_pipeline,

    param_grid=param_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1,

    verbose=2

)


grid_search.fit(

    X_train,

    y_train

)


# ============================================================
# STEP 29: BEST PARAMETERS
# ============================================================

print("\n")
print("=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)

print(
    grid_search.best_params_
)


print("\nBest CV F1 Score:")

print(
    grid_search.best_score_
)


# ============================================================
# STEP 30: FINAL MODEL
# ============================================================

final_model = grid_search.best_estimator_


print("\n")
print("=" * 70)
print("FINAL MODEL CREATED")
print("=" * 70)

print(final_model)


# ============================================================
# STEP 31: FINAL TEST PREDICTION
# ============================================================

final_prediction = final_model.predict(
    X_test
)


# ============================================================
# STEP 32: FINAL METRICS
# ============================================================

final_accuracy = accuracy_score(

    y_test,

    final_prediction

)


final_precision = precision_score(

    y_test,

    final_prediction,

    zero_division=0

)


final_recall = recall_score(

    y_test,

    final_prediction,

    zero_division=0

)


final_f1 = f1_score(

    y_test,

    final_prediction,

    zero_division=0

)


print("\n")
print("=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    "Accuracy :",
    round(final_accuracy, 4)
)

print(
    "Precision:",
    round(final_precision, 4)
)

print(
    "Recall   :",
    round(final_recall, 4)
)

print(
    "F1 Score :",
    round(final_f1, 4)
)


# ============================================================
# STEP 33: CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(

        y_test,

        final_prediction,

        zero_division=0

    )

)


# ============================================================
# STEP 34: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    final_prediction

)


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


plt.figure(

    figsize=(6, 5)

)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "No Churn",
        "Churn"
    ],

    yticklabels=[
        "No Churn",
        "Churn"
    ]

)


plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 35: ROC-AUC
# ============================================================

if hasattr(

    final_model,

    "predict_proba"

):

    probability = final_model.predict_proba(

        X_test

    )[:, 1]


    roc_auc = roc_auc_score(

        y_test,

        probability

    )


    print("\n")
    print("=" * 70)
    print("ROC-AUC SCORE")
    print("=" * 70)

    print(
        round(
            roc_auc,
            4
        )
    )


# ============================================================
# STEP 36: SAVE MODEL
# ============================================================

print("\n")
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)


# IMPORTANT:
#
# We are saving the COMPLETE PIPELINE.
#
# That means:
#
# Input
#   ↓
# Numerical preprocessing
#   ↓
# Categorical preprocessing
#   ↓
# Encoding
#   ↓
# Scaling
#   ↓
# Model
#
# Everything is stored inside model.pkl.


with open(

    MODEL_PATH,

    "wb"

) as file:

    pickle.dump(

        final_model,

        file

    )


print(
    "\nModel saved successfully!"
)

print(
    "Location:"
)

print(
    MODEL_PATH
)


# ============================================================
# STEP 37: LOAD MODEL AGAIN
# ============================================================

print("\n")
print("=" * 70)
print("TESTING SAVED MODEL")
print("=" * 70)


with open(

    MODEL_PATH,

    "rb"

) as file:

    loaded_model = pickle.load(
        file
    )


print(
    "model.pkl loaded successfully."
)


# ============================================================
# STEP 38: TEST SAVED MODEL
# ============================================================

saved_prediction = loaded_model.predict(

    X_test

)


saved_accuracy = accuracy_score(

    y_test,

    saved_prediction

)


print(
    "\nAccuracy after loading model.pkl:"
)

print(
    round(
        saved_accuracy,
        4
    )
)


# ============================================================
# STEP 39: TEST REAL CUSTOMER EXAMPLE
# ============================================================

print("\n")
print("=" * 70)
print("TESTING REAL CUSTOMER INPUT")
print("=" * 70)


# This example is intentionally similar to
# the data coming from your Flask prediction page.


example_customer = pd.DataFrame({

    "Age": [32],

    "Gender": ["Female"],

    "Tenure": [12],

    "Usage Frequency": [18],

    "Support Calls": [3],

    "Payment Delay": [5],

    "Subscription Type": ["Standard"],

    "Contract Length": ["Monthly"],

    "Total Spend": [2500],

    "Last Interaction": [7]

})


print("\nExample customer:")

print(example_customer)


example_prediction = loaded_model.predict(

    example_customer

)


print("\nPrediction:")

print(
    example_prediction[0]
)


# ============================================================
# STEP 40: TEST PREDICTION PROBABILITY
# ============================================================

if hasattr(

    loaded_model,

    "predict_proba"

):

    example_probability = (

        loaded_model.predict_proba(

            example_customer

        )[0][1]

        * 100

    )


    print(
        "\nChurn probability:"
    )

    print(
        round(
            example_probability,
            2
        ),
        "%"
    )


# ============================================================
# STEP 41: CHECK MODEL FEATURES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING MODEL FEATURES")
print("=" * 70)


if hasattr(

    loaded_model,

    "feature_names_in_"

):

    print(
        "\nModel expects these features:"
    )

    print(
        list(
            loaded_model.feature_names_in_
        )
    )


# ============================================================
# STEP 42: FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nYour new model.pkl is ready."
)


print(
    "\nModel location:"
)


print(
    MODEL_PATH
)


print("\n")


print(
    "The Flask application must send exactly these 10 features:"
)


for feature in FEATURE_COLUMNS:

    print(
        " -",
        feature
    )


print("\n")


print("=" * 70)
print("READY FOR FLASK")
print("=" * 70)