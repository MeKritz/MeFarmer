from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML models
waiting_time_model = joblib.load("waiting_time_model.joblib")
workload_model = joblib.load("workload_model.joblib")


# The exact features used while training
FEATURES = [
    "farmers_scheduled_that_day",
    "centre_max_capacity",
    "staff_count",
    "avg_processing_time_min",
    "registered_farmers_nearby",
    "historical_avg_volume_kg",
    "day_of_week",
    "crop_type",
    "time_of_day_slot",
    "season"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from webpage
    farmers_scheduled = int(
        request.form["farmers_scheduled"]
    )

    centre_capacity = int(
        request.form["centre_capacity"]
    )

    staff_count = int(
        request.form["staff_count"]
    )

    processing_time = float(
        request.form["processing_time"]
    )

    registered_farmers = int(
        request.form["registered_farmers"]
    )

    historical_volume = float(
        request.form["historical_volume"]
    )

    day_of_week = int(
        request.form["day_of_week"]
    )

    crop_type = request.form["crop_type"]

    time_of_day = request.form["time_of_day"]

    season = request.form["season"]


    # Create input DataFrame
    input_data = pd.DataFrame([{

        "farmers_scheduled_that_day":
            farmers_scheduled,

        "centre_max_capacity":
            centre_capacity,

        "staff_count":
            staff_count,

        "avg_processing_time_min":
            processing_time,

        "registered_farmers_nearby":
            registered_farmers,

        "historical_avg_volume_kg":
            historical_volume,

        "day_of_week":
            day_of_week,

        "crop_type":
            crop_type,

        "time_of_day_slot":
            time_of_day,

        "season":
            season

    }])


    # -----------------------------
    # AI PREDICTIONS
    # -----------------------------

    predicted_waiting_time = (
        waiting_time_model.predict(input_data)[0]
    )

    predicted_workload = (
        workload_model.predict(input_data)[0]
    )


    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------

    return render_template(
        "index.html",

        waiting_time=round(
            predicted_waiting_time, 2
        ),

        workload=predicted_workload,

        farmer_id="F1025",

        crop=crop_type,

        status="Pending"
    )


if __name__ == "__main__":
    app.run(debug=False)