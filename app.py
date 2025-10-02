import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# --- 1. Load the saved model and pre-processing objects ---
# We use st.cache_resource to load these only once and prevent reloading on every interaction
@st.cache_resource
def load_objects():
    """Loads the trained model, scaler, and power transformer."""
    model = joblib.load('lgbm_cancellation_predictor_model.joblib')
    scaler = joblib.load('scaler.joblib')
    transformer = joblib.load('power_transformer.joblib')
    return model, scaler, transformer

model, scaler, transformer = load_objects()

# This is the list of all columns the model was trained on. 
# It's crucial for creating the final DataFrame for prediction.
TRAINING_COLUMNS = [
    'lead_time', 'no_of_special_requests', 'avg_price_per_room',
    'no_of_adults', 'no_of_weekend_nights', 'required_car_parking_space',
    'no_of_week_nights', 'arrival_month', 'arrival_weekday', 'Total_nights',
    'is_weekend_stay', 'market_segment_type_Complementary',
    'market_segment_type_Corporate', 'market_segment_type_Offline',
    'market_segment_type_Online', 'booking_window_1-3 Months',
    'booking_window_3-12 Months', 'booking_window_Last Week',
    'booking_window_More than a Year', 'party_type_Group',
    'party_type_Solo'
]

# --- 2. App Interface ---
st.set_page_config(layout="wide")
st.title("🏨 INN Hotels Group: Booking Cancellation Predictor")
st.markdown("Enter the details of a booking to predict whether it is likely to be canceled.")

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Booking Details")
    lead_time = st.slider("Lead Time (days)", 0, 450, 50)
    market_segment_type = st.selectbox(
        "Market Segment",
        options=['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation']
    )
    no_of_special_requests = st.slider("Number of Special Requests", 0, 5, 1)

with col2:
    st.header("Stay Details")
    arrival_date = st.date_input("Arrival Date", datetime.date.today())
    no_of_week_nights = st.slider("Weekday Nights", 0, 17, 2)
    no_of_weekend_nights = st.slider("Weekend Nights", 0, 7, 1)
    
with col3:
    st.header("Guest & Room Details")
    no_of_adults = st.slider("Number of Adults", 1, 4, 2)
    avg_price_per_room = st.slider("Average Price per Room ($)", 0, 550, 100)
    required_car_parking_space = st.checkbox("Car Parking Required?")


# --- 3. Prediction Logic ---
if st.button("Predict Cancellation", type="primary", use_container_width=True):
    # Step A: Collect inputs and create a dictionary
    input_data = {
        'lead_time': lead_time,
        'market_segment_type': market_segment_type,
        'no_of_special_requests': no_of_special_requests,
        'avg_price_per_room': avg_price_per_room,
        'no_of_adults': no_of_adults,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'required_car_parking_space': 1 if required_car_parking_space else 0,
        'arrival_date': arrival_date
    }
    
    # Step B: Feature Engineering from the inputs
    df = pd.DataFrame([input_data])
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    
    df['arrival_month'] = df['arrival_date'].dt.month
    df['arrival_weekday'] = df['arrival_date'].dt.weekday
    df['Total_nights'] = df['no_of_week_nights'] + df['no_of_weekend_nights']
    df['is_weekend_stay'] = (df['no_of_weekend_nights'] > 0).astype(int)

    # Binning for 'booking_window'
    bins = [-1, 7, 30, 90, 365, float('inf')]
    labels = ['Last Week', 'Within a Month', '1-3 Months', '3-12 Months', 'More than a Year']
    df['booking_window'] = pd.cut(df['lead_time'], bins=bins, labels=labels)

    # Mapping for 'party_type'
    def get_party_type(n_adults):
        if n_adults == 1: return 'Solo'
        elif n_adults == 2: return 'Couple'
        else: return 'Group'
    df['party_type'] = df['no_of_adults'].apply(get_party_type)
    
    # Drop the original date column
    df = df.drop(columns=['arrival_date'])

    # Step C: One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=[
        'market_segment_type', 'booking_window', 'party_type'
    ])
    
    # Align columns with the training data
    df_aligned = df_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    
    # Identify numerical columns for transformation (must match training)
    numerical_cols = [
        'lead_time', 'no_of_special_requests', 'avg_price_per_room', 
        'no_of_adults', 'no_of_weekend_nights', 'no_of_week_nights',
        'arrival_month', 'arrival_weekday', 'Total_nights'
    ]

    # Step D: Apply Transformations
    df_transformed = transformer.transform(df_aligned[numerical_cols])
    df_scaled = scaler.transform(df_transformed)
    
    # Put scaled numerical data back into the DataFrame
    df_aligned[numerical_cols] = df_scaled

    # Step E: Make Prediction
    prediction_proba = model.predict_proba(df_aligned)[:, 1]
    cancellation_probability = prediction_proba[0]
    
    # The optimal threshold for your model (344) was 0.40
    OPTIMAL_THRESHOLD = 0.40
    
    # --- 4. Display Results ---
    st.markdown("---")
    st.header("Prediction Result")
    
    if cancellation_probability >= OPTIMAL_THRESHOLD:
        st.error(f"🔴 **High Risk of Cancellation**", icon="🚨")
    else:
        st.success(f"🟢 **Low Risk of Cancellation**", icon="✅")

    st.subheader(f"Cancellation Probability: {cancellation_probability:.2%}")
    st.progress(cancellation_probability)

    with st.expander("Why this prediction?"):
        st.markdown(f"""
        This prediction is based on a LightGBM model trained on historical booking data. 
        The model identified several key factors influencing cancellations:
        
        - **High lead time ({lead_time} days):** Bookings made far in advance have more time for plans to change.
        - **Higher average price (${avg_price_per_room}/night):** More expensive bookings might be more sensitive to budget changes.
        - **Market segment ('{market_segment_type}'):** Online bookings, for example, often have higher cancellation rates due to flexible policies.
        
        **Business Recommendation:** For high-risk bookings, consider a proactive approach such as a confirmation email a week before arrival or offering a small, non-refundable discount to lock in the reservation.
        """)

