# ============================================================
# CHURNAI - CUSTOMER CHURN PREDICTION
# COMPLETE FLASK APPLICATION
# ============================================================

from pathlib import Path
from datetime import datetime
import os
import pickle
import sqlite3

import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)


# ============================================================
# STEP 1: FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# STEP 2: PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"

DATABASE_PATH = BASE_DIR / "predictions.db"


# ============================================================
# STEP 3: MODEL FEATURES
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


# ============================================================
# STEP 4: NUMERICAL FEATURES
# ============================================================

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
# STEP 5: LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("CHURNAI - FLASK APPLICATION")
print("=" * 60)

print()
print("Loading model...")

try:

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)

except Exception as error:

    model = None

    print()
    print("ERROR: Model could not be loaded.")
    print("Reason:", error)


# ============================================================
# STEP 6: DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# STEP 7: CREATE DATABASE
# ============================================================

def initialize_database():

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            age REAL NOT NULL,

            gender TEXT NOT NULL,

            tenure REAL NOT NULL,

            usage_frequency REAL NOT NULL,

            support_calls REAL NOT NULL,

            payment_delay REAL NOT NULL,

            subscription_type TEXT NOT NULL,

            contract_length TEXT NOT NULL,

            total_spend REAL NOT NULL,

            last_interaction REAL NOT NULL,

            prediction INTEGER NOT NULL,

            probability REAL NOT NULL,

            risk TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()

    print()
    print("Database ready.")
    print("Database:", DATABASE_PATH)


# Initialize database when application starts
initialize_database()


# ============================================================
# STEP 8: HOME / PREDICTION PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# STEP 9: DASHBOARD PAGE
# ============================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# STEP 10: ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# STEP 11: CHECK MODEL
# ============================================================

@app.route("/health")
def health():

    if model is None:

        return jsonify({
            "success": False,
            "model_loaded": False,
            "message": "Model is not loaded."
        }), 500


    return jsonify({
        "success": True,
        "model_loaded": True,
        "message": "ChurnAI model is ready.",
        "features": FEATURE_COLUMNS
    })


# ============================================================
# STEP 12: HELPER - CONVERT NUMBER
# ============================================================

def convert_number(
    data,
    column
):

    value = data.get(column)

    if value is None:
        raise ValueError(
            f"Missing field: {column}"
        )

    if str(value).strip() == "":
        raise ValueError(
            f"Empty field: {column}"
        )

    try:

        return float(value)

    except (ValueError, TypeError):

        raise ValueError(
            f"{column} must be a valid number."
        )


# ============================================================
# STEP 13: GET CHURN PROBABILITY
# ============================================================

def get_churn_probability(
    input_data,
    prediction
):

    # --------------------------------------------------------
    # If model supports predict_proba
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = list(
            getattr(
                model,
                "classes_",
                []
            )
        )


        # ----------------------------------------------------
        # Find class 1
        # ----------------------------------------------------

        churn_index = None

        for index, class_value in enumerate(classes):

            if str(class_value).strip().lower() in {
                "1",
                "yes",
                "true",
                "churn",
                "positive"
            }:

                churn_index = index

                break


        # ----------------------------------------------------
        # If class 1 was not found
        # ----------------------------------------------------

        if churn_index is None:

            # If prediction itself is numeric 1
            if str(prediction).strip() == "1":

                for index, class_value in enumerate(classes):

                    if str(class_value).strip() == "1":

                        churn_index = index

                        break


        # ----------------------------------------------------
        # Final fallback
        # ----------------------------------------------------

        if churn_index is None:

            if len(probabilities) >= 2:

                churn_index = 1

            else:

                churn_index = 0


        probability = (
            float(
                probabilities[churn_index]
            ) * 100
        )


        return round(
            probability,
            2
        )


    # --------------------------------------------------------
    # Model does not support predict_proba
    # --------------------------------------------------------

    prediction_string = str(
        prediction
    ).strip().lower()


    if prediction_string in {
        "1",
        "yes",
        "true",
        "churn",
        "positive"
    }:

        return 100.0


    return 0.0


# ============================================================
# STEP 14: CALCULATE RISK
# ============================================================

def calculate_risk(
    probability
):

    if probability >= 70:

        return "High"

    elif probability >= 40:

        return "Medium"

    else:

        return "Low"


# ============================================================
# STEP 15: PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({
                "success": False,
                "error": "Model is not loaded."
            }), 500


        # ----------------------------------------------------
        # Get JSON data
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "success": False,
                "error": "No prediction data received."
            }), 400


        print()
        print("=" * 60)
        print("NEW PREDICTION REQUEST")
        print("=" * 60)

        print(data)


        # ----------------------------------------------------
        # Check required fields
        # ----------------------------------------------------

        missing_fields = []

        for column in FEATURE_COLUMNS:

            if (
                column not in data
                or data[column] is None
                or str(data[column]).strip() == ""
            ):

                missing_fields.append(
                    column
                )


        if missing_fields:

            return jsonify({
                "success": False,
                "error": "Missing fields.",
                "missing_fields": missing_fields
            }), 400


        # ----------------------------------------------------
        # Prepare numerical values
        # ----------------------------------------------------

        age = convert_number(
            data,
            "Age"
        )

        tenure = convert_number(
            data,
            "Tenure"
        )

        usage_frequency = convert_number(
            data,
            "Usage Frequency"
        )

        support_calls = convert_number(
            data,
            "Support Calls"
        )

        payment_delay = convert_number(
            data,
            "Payment Delay"
        )

        total_spend = convert_number(
            data,
            "Total Spend"
        )

        last_interaction = convert_number(
            data,
            "Last Interaction"
        )


        # ----------------------------------------------------
        # Prepare categorical values
        # ----------------------------------------------------

        gender = str(
            data["Gender"]
        ).strip()

        subscription_type = str(
            data["Subscription Type"]
        ).strip()

        contract_length = str(
            data["Contract Length"]
        ).strip()


        # ----------------------------------------------------
        # Create DataFrame
        #
        # IMPORTANT:
        # Column names and order match training.
        # ----------------------------------------------------

        input_row = {

            "Age": age,

            "Gender": gender,

            "Tenure": tenure,

            "Usage Frequency": usage_frequency,

            "Support Calls": support_calls,

            "Payment Delay": payment_delay,

            "Subscription Type": subscription_type,

            "Contract Length": contract_length,

            "Total Spend": total_spend,

            "Last Interaction": last_interaction

        }


        input_dataframe = pd.DataFrame(
            [input_row],
            columns=FEATURE_COLUMNS
        )


        print()
        print("Input DataFrame:")
        print(input_dataframe)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction_raw = model.predict(
            input_dataframe
        )[0]


        # ----------------------------------------------------
        # Convert prediction to 0 / 1
        # ----------------------------------------------------

        try:

            prediction = int(
                prediction_raw
            )

        except (ValueError, TypeError):

            prediction_text = str(
                prediction_raw
            ).strip().lower()


            if prediction_text in {
                "1",
                "yes",
                "true",
                "churn",
                "positive"
            }:

                prediction = 1

            else:

                prediction = 0


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = get_churn_probability(
            input_dataframe,
            prediction
        )


        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        risk = calculate_risk(
            probability
        )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        now = datetime.now()

        timestamp_database = now.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        timestamp_display = now.strftime(
            "%d %b %Y, %I:%M %p"
        )


        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        connection = get_db_connection()

        cursor = connection.cursor()


        cursor.execute(
            """
            INSERT INTO predictions (

                timestamp,

                age,
                gender,
                tenure,
                usage_frequency,
                support_calls,
                payment_delay,
                subscription_type,
                contract_length,
                total_spend,
                last_interaction,

                prediction,
                probability,
                risk

            )

            VALUES (

                ?,

                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,

                ?,
                ?,
                ?

            )
            """,
            (

                timestamp_database,

                age,
                gender,
                tenure,
                usage_frequency,
                support_calls,
                payment_delay,
                subscription_type,
                contract_length,
                total_spend,
                last_interaction,

                prediction,
                probability,
                risk

            )
        )


        prediction_id = cursor.lastrowid


        connection.commit()

        connection.close()


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        print()
        print("Prediction:", prediction)
        print("Probability:", probability)
        print("Risk:", risk)
        print("Database ID:", prediction_id)


        print()
        print("=" * 60)
        print("PREDICTION COMPLETED")
        print("=" * 60)


        return jsonify({

            "success": True,

            "id": prediction_id,

            "prediction": prediction,

            "probability": probability,

            "risk": risk,

            "timestamp": timestamp_display

        })


    except Exception as error:

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(error)


        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# STEP 16: DASHBOARD API
# ============================================================

@app.route(
    "/api/dashboard",
    methods=["GET"]
)
def dashboard_api():

    try:

        connection = get_db_connection()

        cursor = connection.cursor()


        # ----------------------------------------------------
        # GET ALL RECORDS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """
        )


        rows = cursor.fetchall()


        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        total = len(rows)


        # ----------------------------------------------------
        # RISK COUNTS
        # ----------------------------------------------------

        high = 0

        medium = 0

        low = 0


        # ----------------------------------------------------
        # AVERAGE PROBABILITY
        # ----------------------------------------------------

        probabilities = []


        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history = []


        for row in rows:

            risk = row["risk"]


            if risk == "High":

                high += 1

            elif risk == "Medium":

                medium += 1

            else:

                low += 1


            probabilities.append(
                float(
                    row["probability"]
                )
            )


            history.append({

                "id": row["id"],

                "timestamp":
                    datetime.strptime(
                        row["timestamp"],
                        "%Y-%m-%d %H:%M:%S"
                    ).strftime(
                        "%d %b %Y, %I:%M %p"
                    ),

                "prediction":
                    row["prediction"],

                "probability":
                    row["probability"],

                "risk":
                    row["risk"],

                "customer": {

                    "Age":
                        row["age"],

                    "Gender":
                        row["gender"],

                    "Tenure":
                        row["tenure"],

                    "Usage Frequency":
                        row["usage_frequency"],

                    "Support Calls":
                        row["support_calls"],

                    "Payment Delay":
                        row["payment_delay"],

                    "Subscription Type":
                        row["subscription_type"],

                    "Contract Length":
                        row["contract_length"],

                    "Total Spend":
                        row["total_spend"],

                    "Last Interaction":
                        row["last_interaction"]

                }

            })


        # ----------------------------------------------------
        # AVERAGE
        # ----------------------------------------------------

        if probabilities:

            average_probability = (
                sum(probabilities)
                / len(probabilities)
            )

        else:

            average_probability = 0


        # ----------------------------------------------------
        # LATEST
        # ----------------------------------------------------

        latest = (
            history[0]
            if history
            else None
        )


        connection.close()


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "stats": {

                "total":
                    total,

                "high":
                    high,

                "medium":
                    medium,

                "low":
                    low,

                "average_probability":
                    round(
                        average_probability,
                        2
                    )

            },

            "latest":
                latest,

            "history":
                history

        })


    except Exception as error:

        print(
            "Dashboard error:",
            error
        )


        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# STEP 17: DELETE ONE PREDICTION
# ============================================================

@app.route(
    "/delete-prediction/<int:prediction_id>",
    methods=["DELETE"]
)
def delete_prediction(
    prediction_id
):

    try:

        connection = get_db_connection()

        cursor = connection.cursor()


        # ----------------------------------------------------
        # CHECK RECORD
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM predictions
            WHERE id = ?
            """,
            (
                prediction_id,
            )
        )


        record = cursor.fetchone()


        if record is None:

            connection.close()


            return jsonify({

                "success": False,

                "error":
                    "Prediction record not found."

            }), 404


        # ----------------------------------------------------
        # DELETE
        # ----------------------------------------------------

        cursor.execute(
            """
            DELETE FROM predictions
            WHERE id = ?
            """,
            (
                prediction_id,
            )
        )


        connection.commit()

        connection.close()


        print(
            f"Prediction {prediction_id} deleted."
        )


        return jsonify({

            "success": True,

            "message":
                "Prediction deleted successfully."

        })


    except Exception as error:

        print(
            "Delete error:",
            error
        )


        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# STEP 18: CLEAR ALL PREDICTION HISTORY
# ============================================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    try:

        connection = get_db_connection()

        cursor = connection.cursor()


        cursor.execute(
            """
            DELETE FROM predictions
            """
        )


        deleted_count = cursor.rowcount


        connection.commit()

        connection.close()


        print(
            f"Cleared {deleted_count} prediction records."
        )


        return jsonify({

            "success": True,

            "message":
                "All prediction history cleared.",

            "deleted_count":
                deleted_count

        })


    except Exception as error:

        print(
            "Clear history error:",
            error
        )


        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# STEP 19: RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CHURNAI SERVER")
    print("=" * 60)

    print()
    print("Local URL:")
    print("http://127.0.0.1:5000")

    print()
    print("Network URL:")
    print("http://0.0.0.0:5000")

    print()
    print("Dashboard:")
    print("http://127.0.0.1:5000/dashboard")

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Database:")
    print(DATABASE_PATH)

    print()
    print("=" * 60)


    # AWS-compatible port
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    # Debug can be enabled locally using:
    # set FLASK_DEBUG=1
    debug_mode = (
        os.environ.get(
            "FLASK_DEBUG",
            "0"
        ) == "1"
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=debug_mode

    )