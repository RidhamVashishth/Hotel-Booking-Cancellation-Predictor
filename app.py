import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib

# Load model and transformer
with open('final_model.joblib', 'rb') as file:
    model = joblib.load(file)

with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)

# Helper function to safely convert input
def safe_float(val):
    try:
        return float(val)
    except:
        return None

def safe_int(val):
    try:
        return int(val)
    except:
        return None

# Prediction logic
def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]

    if pred > 0.5:
        return f'This booking is more likely to get canceled, chances {round(pred, 2)}'
    else:
        return f'This booking is less likely to get canceled, chances {round(pred, 2)}'

# Streamlit UI
def main():
    st.title('INN HOTEL GROUP')

    # User input fields (clean, empty by default)
    lt = st.text_input('Enter the lead time in days')
    price = st.text_input('Enter the price of the room')
    weekn = st.text_input('Enter number of week nights')
    wkndn = st.text_input('Enter number of weekend nights')

    mkt = 1 if st.selectbox('How the booking was made', ['Select', 'Online', 'Offline']) == 'Online' else 0
    adult = st.selectbox('How many adults?', ['Select', 1, 2, 3, 4])
    # arr_m = st.selectbox('Month of arrival?', ['Select'] + list(range(1, 13)))
    arr_m = st.slider('What is the month of arrival?', min_value=1, max_value=12, step=1)

    weekday_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thus': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    arr_day = st.selectbox('Arrival weekday', ['Select'] + list(weekday_map.keys()))
    dep_day = st.selectbox('Departure weekday', ['Select'] + list(weekday_map.keys()))

    park = st.selectbox('Need parking?', ['Select', 'yes', 'no'])
    park = 1 if park == 'yes' else 0 if park == 'no' else None

    spcl = st.selectbox('Number of special requests', ['Select', 0, 1, 2, 3, 4, 5])

    # Validate inputs
    lt_val = safe_float(lt)
    price_val = safe_float(price)
    weekn_val = safe_int(weekn)
    wkndn_val = safe_int(wkndn)

    if st.button('Predict'):
        if None in [lt_val, price_val, weekn_val, wkndn_val] or \
           'Select' in [adult, arr_m, arr_day, dep_day, spcl] or park is None:
            st.warning("⚠️ Please fill in all fields correctly before predicting.")
            return

        try:
            # Transform lead time and price
            lt_t, price_t = transformer.transform([[lt_val, price_val]])[0]

            # Convert dropdown values
            arr_w = weekday_map[arr_day]
            dep_w = weekday_map[dep_day]
            totan = weekn_val + wkndn_val

            # Prepare feature list
            inp_list = [lt_t, spcl, price_t, adult, wkndn_val, park, weekn_val, mkt, arr_m, arr_w, totan, dep_w]

            response = prediction(inp_list)
            st.success(response)

        except Exception as e:
            st.error(f" Prediction failed: {str(e)}")

# Run the app
if __name__ == '__main__':
    main()
