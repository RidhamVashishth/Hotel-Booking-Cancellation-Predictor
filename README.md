# **Hotel Booking Cancellation Prediction**

## **1\. Project Overview**

This project presents a data-driven solution to predict hotel booking cancellations for the INN Hotels Group. By leveraging machine learning, the system identifies bookings with a high likelihood of being canceled, enabling the hotel to implement proactive strategies to mitigate revenue loss and improve operational efficiency. The final model is deployed as an interactive web application using Streamlit.

👉 [Click here to access the live app](https://innhotel-booking-cancellation-predictor.streamlit.app/)

## **2\. Business Problem**

The INN Hotels Group has been facing significant financial challenges due to a rising number of booking cancellations, which recently peaked at an 18% inventory loss. This translates to an approximate annual revenue loss of $0.25 million. The previous heuristic-based methods for predicting cancellations have proven ineffective and unscalable.

The core problem is the inability to anticipate which bookings will be canceled, leading to:

* **Lost Revenue:** Empty rooms that could have been resold.  
* **Reduced Profit Margins:** Last-minute price reductions to attract new customers.  
* **Operational Inefficiency:** Difficulty in managing staff, inventory, and overbooking strategies.

This project aims to replace the outdated system with a robust machine learning model that provides a reliable cancellation probability for each booking.

## **3\. Data Science Solution**

A comprehensive machine learning pipeline was developed to address the business problem. The process involved rigorous data analysis, feature engineering, and a systematic model evaluation framework to select the most effective algorithm for the task.

The final solution is a **LightGBM (Light Gradient Boosting Machine)** model, which demonstrated a superior ability to identify potential cancellations. This model was trained on historical booking data and is served through a user-friendly Streamlit interface where hotel managers can input new booking details and receive an instant cancellation prediction.

## **4\. Methodology**

The project followed a structured data science lifecycle:

1. **Data Exploration & Cleaning:** The initial dataset was analyzed to understand variable distributions, identify outliers, and check for missing values. Statistical tests (Mann-Whitney U, Chi-square) were used to validate the relationships between features and the target variable (booking\_status).  
2. **Feature Engineering:** New, informative features were created to enhance the model's predictive power by capturing underlying patterns in the booking data. The goal was to translate raw data into signals that are more directly related to cancellation behavior.  
   * **Date-Based Features:** arrival\_month and arrival\_weekday were extracted from the arrival date. This helps the model capture seasonality (e.g., cancellations might be higher during peak holiday months) and weekly patterns (e.g., weekend vs. weekday arrivals).  
   * **Stay Duration Features:** Total\_nights was calculated by summing weekend and week nights. Longer stays might have a different cancellation profile than shorter ones. A binary is\_weekend\_stay feature was also created to specifically capture bookings that include high-demand weekend periods.  
   * **Customer Segmentation Features:** A party\_type feature (Solo, Couple, Group) was engineered from the number of adults. This helps segment customers, as a large corporate group might have different cancellation reasons and probabilities than a solo traveler.  
3. **Data Preprocessing:** The data was meticulously prepared for modeling through:  
   * **Outlier Capping:** Using a 2.0x IQR multiplier to handle extreme values.  
   * **Encoding:** Converting categorical features to a numerical format using one-hot encoding.  
   * **Transformation & Scaling:** Applying PowerTransformer to normalize data distributions and StandardScaler to scale features.  
4. **Model Building & Evaluation:**  
   * A wide range of models were tested, including Logistic Regression, LDA, Random Forest, XGBoost, CatBoost, and LightGBM.  
   * The impact of data sampling techniques (SMOTE, UnderSampler) was evaluated.  
   * The primary evaluation metrics were **Recall(1)** and **F2-Score(1)**, as the business cost of a missed cancellation (False Negative) is significantly higher than a false alarm (False Positive).  
5. **Hyperparameter Tuning:** RandomizedSearchCV and GridSearchCV were used to find the optimal hyperparameters for each model, maximizing their performance.  
6. **Final Model Selection:** The **LightGBM model, trained on an under-sampled dataset**, was chosen as the final model. While several models (like XGBoost on the original data) achieved high performance, the LightGBM with UnderSampler was selected for several key reasons:  
   * **Superior Balance:** It provided an excellent balance between high **Recall** (catching a large percentage of actual cancellations) and reasonable **Precision**. This ensures the hotel can act on a high number of at-risk bookings without overwhelming staff with too many false alarms.  
   * **Focus on the Minority Class:** Under-sampling the majority class ('Not Canceled') forces the model to pay closer attention to the minority class ('Canceled'), which is the primary target of our business problem. This resulted in a model that is highly sensitive to the patterns of cancellations.  
   * **Efficiency:** LightGBM is known for its high training speed and lower memory usage compared to other gradient-boosting models. This makes it ideal for a deployed application where quick predictions and efficient resource use are important.

## **5\. Technology Stack**

* **Programming Language:** Python 3.10+  
* **Data Manipulation & Analysis:** Pandas, NumPy  
* **Machine Learning:** Scikit-learn, LightGBM  
* **Web Framework:** Streamlit  
* **Data Sampling:** Imbalanced-learn

## **6\. Project Structure**

├── lgbm\_cancellation\_model.joblib  \# Saved final ML model  
├── power\_transformer.joblib        \# Saved PowerTransformer object  
├── scaler.joblib                   \# Saved StandardScaler object  
├── app.py                          \# The Streamlit application script  
├── requirements.txt                \# Project dependencies  
└── README.md                       \# Project documentation  
