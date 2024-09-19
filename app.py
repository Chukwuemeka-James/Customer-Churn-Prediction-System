import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')

# Add a sidebar for instructions and explanation
st.sidebar.title('Instructions')
st.sidebar.info(
    """
    This app predicts whether a customer will churn based on their profile.
    Please fill in the customer details on the right.
    """
)

# User input fields
st.header('Customer Details')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, help='Select customer age')
balance = st.number_input('Balance', min_value=0.0, max_value=1e6, step=1000.0, help='Customer bank balance')
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, step=1, help='Customer credit score')
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=1e6, step=1000.0, help='Customer estimated salary')
tenure = st.slider('Tenure (Years)', 0, 10, help='Number of years the customer has been with the bank')
num_of_products = st.slider('Number of Products', 1, 4, help='Number of products the customer has purchased')
has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display result with improved formatting
st.subheader('Churn Prediction Result')
st.write(f'Churn Probability: **{prediction_proba:.2f}**')

# Display a progress bar to visualize the probability
st.progress(int(prediction_proba * 100))

# Show prediction message
if prediction_proba > 0.5:
    st.error('The customer is **likely to churn**.')
else:
    st.success('The customer is **not likely to churn**.')

# Additional visualization (optional)
st.subheader('Probability Distribution')
fig, ax = plt.subplots()
ax.bar(['Not Churn', 'Churn'], [1 - prediction_proba, prediction_proba], color=['green', 'red'])
st.pyplot(fig)

# Provide more insights based on the churn probability
if prediction_proba > 0.5:
    st.write('Consider actions like offering better rewards or personalized services to retain the customer.')
else:
    st.write('The customer is engaged, continue with your current strategies.')
