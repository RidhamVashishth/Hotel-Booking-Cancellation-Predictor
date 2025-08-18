# 🏨 Hotel Booking Cancellation Prediction App

This is a **Machine Learning-powered web application** built with **Streamlit** that predicts whether a hotel booking is likely to be canceled or not. It's designed to help hotel managers, revenue analysts, and travel businesses make smarter decisions based on customer and booking attributes.

---

## 🚀 Live Demo

👉 **[Click here to access the live app](https://hotel-cancellation-predictionn.streamlit.app/)**

---

## 📌 Features

- Predicts the **likelihood of booking cancellation** using historical data.
- Clean, intuitive **Streamlit user interface**.
- Automatically handles **data transformation and model predictions**.
- Input validations for better **user experience and reliability**.
- Gives probabilistic output (e.g., "There is a 78% chance of cancellation").

---

## 📊 Dataset

The dataset used for training the model is based on **real-world hotel booking records**, similar to the popular public dataset available on [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). It includes features like:

- Lead time  
- Room price  
- Number of adults  
- Arrival/departure weekday  
- Number of weekend/week nights  
- Special requests  
- Parking availability  
- Booking source (online/offline)  
- ...and more

*Note: The dataset was cleaned, preprocessed, and transformed before training the model.*

---

## 🧠 Model & Tech Stack

- **Machine Learning Model:** Trained using `sklearn` with logistic regression (or similar classification algorithm)
- **Preprocessing:** Custom transformers for numeric scaling and feature encoding
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Model Deployment:** Hosted via Streamlit Cloud

---

## 🛠 How It Works

1. User enters booking details (lead time, price, arrival info, etc.)
2. Input is transformed using a pre-trained transformer
3. Model returns the probability of cancellation
4. Output is shown as a friendly message

---

## 📂 Files Included

- `app.py`: Main Streamlit application file
- `final_model.joblib`: Trained ML model
- `transformer.pkl`: Preprocessing pipeline
- `README.md`: Project documentation

---


