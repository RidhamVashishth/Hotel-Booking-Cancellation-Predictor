

import numpy as np
import pandas as pd
import streamlit as st
import pickle

with open('final_model.pkl', 'rb') as file:
    model= pickle.load(file)

with open('transformer.pkl', 'rb') as file:
    transformer= pickle.load(file)
def prediction(input_list):
    input_list= np.array(input_list, dtype=object)
    pred= model.predict_proba([input_list])[:,1][0]
    if pred>0.5:
        return f'This booking is more likely to get cancelled, with chances {round(pred,2)}'
    else:
        return f'This booking is less likely to get cancelled, with chances {round(pred,2)}'

def main():
    st.title('Inn Hotel Group')
    lt= st.text_input('Enter the lead time in Days')
    mkt= (lambda x:1 if x=='Online' else 0)(st.select_box('How the booking was made?', ['Online', 'Offline']))
    price= st.text_input('Enter the price of the room ')
    adult=st.selectbox([1,2,3,4])
    arr_m=st.slider('What is the month of arrival?', min_value=1, max_value=12, step=1)
    weekd_lambda=(lambda x:0 if x=='Mon' else 
                          1 if x=='Tue' else
                          2 if x=='Wed' else 
                          3 if x=='Thu' else
                          4 if x=='Fri' else
                          5 if x=='Sat' else
                          6)
    arr_w= weekd_lambda(st.selectbox('What is the weekday or arrival?', ['Mon', 'Tue', 'Wed', 'Thu','Fri', 'Sat', 'Sun']))
    dep_w= weekd_lambda(st.selectbox('What is the weekday or departure?', ['Mon', 'Tue', 'Wed', 'Thu','Fri', 'Sat', 'Sun']))
    weekn= st.text_input('Enter the no of weeknights in stay?')
    wkndn= st.text_input('Enter the no of weekend nights in stay?')
    totan= weekn+wkndn
    park = (lambda x: 1 if x == 'Yes' else 0)(st.selectbox('Does the customer need parking?', ['Yes', 'No']))
    spcl= st.selectbox('How many special requests have been made?', [0,1,2,3,4])
    lt_t, price_t= transformer.transform([[lt, price]])[0]
    inp_list=[lt_t, spcl, price_t, adult, wkndn, park, weekn, mkt, arr_m, arr_w, totan, dep_w]

    if st.button('Predict'):
        response= prediction(inp_list)
        st.success(response)
if __name__=='__main__':
    main()
