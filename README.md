# Customer Churn Prediction

## Overview
This project implements a Customer Churn Prediction model for a financial institution using TensorFlow and Streamlit. The model predicts whether a customer will churn based on various customer attributes. The application allows users to input customer details and get real-time predictions on churn probability.

## Dataset
The dataset used for this project is a customer churn dataset which contains the following features:

- **CreditScore**: Credit score of the customer.
- **Geography**: Country of residence (e.g., France, Spain, Germany).
- **Gender**: Gender of the customer (Male/Female).
- **Age**: Age of the customer.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Account balance of the customer.
- **NumOfProducts**: Number of products purchased by the customer.
- **HasCrCard**: Whether the customer has a credit card (1 for Yes, 0 for No).
- **IsActiveMember**: Whether the customer is an active member (1 for Yes, 0 for No).
- **EstimatedSalary**: Estimated salary of the customer.
- **Exited**: Target variable indicating if the customer has churned (1 for Yes, 0 for No).

### Project Overview

The Customer Churn Prediction project utilizes machine learning to forecast whether customers are likely to leave a service. It involves the following key steps:

1. **Data Preprocessing**: The dataset is cleaned and preprocessed, including the encoding of categorical variables and feature scaling.
2. **Model Training**: An Artificial Neural Network (ANN) is built and trained using TensorFlow, optimizing its performance through techniques like early stopping and TensorBoard for monitoring.
3. **Prediction**: A separate prediction script loads the trained model to evaluate new customer data, providing a probability score for churn.
4. **Deployment**: A Streamlit application is developed for user-friendly interaction, allowing users to input customer details and view predictions.
5. **Visualization**: The app features visual outputs, including churn probability distribution and actionable insights based on the prediction results.
6. **Goal**: The overall goal is to help businesses proactively manage customer relationships by identifying at-risk customers and implementing targeted retention strategies.


## Project Structure
The project consists of the following files:

- `Churn_Modelling.csv`: The dataset used for training and testing the model.
- `experiment.ipynb`: Jupyter notebook for data preprocessing, model training, and saving the model and encoders.
- `Prediction.ipynb`: Jupyter notebook for loading the trained model and making predictions on sample input data.
- `app.py`: Streamlit application for user interaction and churn prediction.

## Installation
To run this project you have to install python and then `pip install requirements.txt`

## Usage
1. Run the Streamlit app:

## Model Training
The model was trained using a feedforward Artificial Neural Network (ANN) implemented in TensorFlow. Key steps in the training process include:
- Data preprocessing: Encoding categorical variables, scaling features, and splitting the dataset into training and test sets.
- Model architecture: The ANN consists of two hidden layers with ReLU activation and one output layer with a sigmoid activation function.
- Training: The model is trained with binary cross-entropy loss and Adam optimizer, with early stopping and TensorBoard for monitoring.