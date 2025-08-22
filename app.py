import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# --- 1. Load Saved Objects ---
# Load the trained model and the power transformer
try:
    model = joblib.load('voting_classifier_model.pkl')
    power_transformer = joblib.load('power_transformer.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading a required file: {e}. Please ensure both voting_classifier_model.pkl and power_transformer.pkl are in the repository.")
    st.stop()

# --- 2. App Title and Description ---
st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨")
st.title('🏨 Hotel Booking Cancellation Predictor')
st.markdown("""
This application predicts whether a hotel booking will be canceled based on the details provided. The prediction is made using a pre-trained machine learning model.
""")

# --- 3. User Input Sidebar ---
st.sidebar.header('Enter Booking Details')

def user_input_features():
    # Create input fields for the user to enter data
    lead_time = st.sidebar.slider('Lead Time (days)', 0, 450, 50, help="Number of days between booking and arrival.")
    market_segment_type = st.sidebar.selectbox('Market Segment Type', ('Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'), help="The channel through which the booking was made.")
    no_of_special_requests = st.sidebar.slider('Number of Special Requests', 0, 5, 1, help="Count of special requests made by the customer (e.g., twin bed).")
    avg_price_per_room = st.sidebar.slider('Average Price Per Room ($)', 0.0, 550.0, 100.0, help="The average daily rate for the room.")
    no_of_adults = st.sidebar.slider('Number of Adults', 1, 4, 2, help="The number of adults included in the booking.")
    no_of_weekend_nights = st.sidebar.slider('Number of Weekend Nights', 0, 7, 1, help="Number of weekend nights (Saturday or Sunday) the guest will stay.")
    arrival_date = st.sidebar.date_input('Arrival Date', datetime.now(), help="The scheduled arrival date for the booking.")
    required_car_parking_space = st.sidebar.selectbox('Required Car Parking Space', (0, 1), help="Does the customer require a car parking space (1 for Yes, 0 for No).")
    no_of_week_nights = st.sidebar.slider('Number of Week Nights', 0, 17, 2, help="Number of week nights (Monday to Friday) the guest will stay.")

    # Assemble the data into a DataFrame
    data = {
        'lead_time': lead_time,
        'market_segment_type': market_segment_type,
        'no_of_special_requests': no_of_special_requests,
        'avg_price_per_room': avg_price_per_room,
        'no_of_adults': no_of_adults,
        'no_of_weekend_nights': no_of_weekend_nights,
        'arrival_date': arrival_date,
        'required_car_parking_space': required_car_parking_space,
        'no_of_week_nights': no_of_week_nights
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- 4. Feature Engineering ---
# This section must exactly replicate the preprocessing from your notebook

# Convert arrival_date to datetime
input_df["arrival_date"] = pd.to_datetime(input_df["arrival_date"])

# One-hot encode market_segment_type
input_df['market_segment_type_Online'] = 1 if input_df['market_segment_type'][0] == 'Online' else 0

# Apply PowerTransformer to numerical columns
num_cols = ["lead_time", "avg_price_per_room"]
input_df[num_cols] = power_transformer.transform(input_df[num_cols])

# Create date-based features
input_df["arrival_month"] = input_df["arrival_date"].dt.month
input_df["arrival_weekday"] = input_df["arrival_date"].dt.weekday

# Create Total_nights feature
input_df["Total_nights"] = input_df["no_of_week_nights"] + input_df["no_of_weekend_nights"]

# Create departure_weekday feature
input_df["departure_weekday"] = input_df["arrival_weekday"] + input_df["Total_nights"]

def setting_weekday(num):
    return num % 7 if num > 6 else num

input_df["departure_weekday"] = input_df["departure_weekday"].apply(setting_weekday)

# --- 5. Final Feature Selection and Ordering ---
# Drop columns that are no longer needed for prediction
input_df = input_df.drop(columns=["arrival_date", "market_segment_type"])

# Define the final expected order of columns based on your notebook's training data
expected_columns = [
    'lead_time', 'no_of_special_requests', 'avg_price_per_room', 'no_of_adults',
    'no_of_weekend_nights', 'required_car_parking_space', 'no_of_week_nights',
    'market_segment_type_Online', 'arrival_month', 'arrival_weekday',
    'Total_nights', 'departure_weekday'
]

# Add any missing columns and fill with 0, then reorder
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0
final_df = input_df[expected_columns]

st.subheader('Review Your Input (After Transformation)')
st.write(final_df)


# --- 6. Prediction and Output ---
if st.sidebar.button('Predict Cancellation'):
    try:
        # Make prediction
        prediction = model.predict(final_df)
        prediction_proba = model.predict_proba(final_df)

        # CORRECTLY Decode the prediction based on {"Canceled":1 , "Not Canceled":0}
        booking_status = "Canceled" if prediction[0] == 1 else "Not Canceled"

        st.subheader('Prediction Result')
        if booking_status == "Canceled":
            st.warning(f"The booking is predicted to be **{booking_status}**.")
        else:
            st.success(f"The booking is predicted to be **{booking_status}**.")

        # CORRECTLY Display probabilities
        st.subheader('Prediction Probability')
        st.write(f"Probability of being 'Not Canceled' (Class 0): **{prediction_proba[0][0]:.2f}**")
        st.write(f"Probability of being 'Canceled' (Class 1): **{prediction_proba[0][1]:.2f}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
