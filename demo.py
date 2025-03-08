# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import pickle as pk
import warnings

warnings.simplefilter("ignore")

# Title of the Streamlit app
st.title('Titanic Survival Prediction: Logistic Regression')
st.sidebar.header('Passenger Information')

# Function to take user input
def user_input_features():
    Pclass = st.sidebar.selectbox('Passenger Class (Pclass)', [1, 2, 3])
    Sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])  # Using string inputs
    Age = st.sidebar.number_input("Insert Age", min_value=0.0, max_value=100.0, value=30.0)
    SibSp = st.sidebar.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, value=0)
    Parch = st.sidebar.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, value=0)
    Fare = st.sidebar.number_input("Fare Amount", min_value=0.0, value=10.0)
    Embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])  # C = Cherbourg, Q = Queenstown, S = Southampton

    # Convert categorical inputs to numerical values
    Sex = 1 if Sex == 'Male' else 0  # Male = 1, Female = 0

    # One-hot encoding for Embarked
    Embarked_Q = 1 if Embarked == 'Q' else 0
    Embarked_S = 1 if Embarked == 'S' else 0

    # Create DataFrame with only the required columns
    data = {
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked_Q': Embarked_Q,
        'Embarked_S': Embarked_S
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Load the trained model
model_path = r"logistic_regression.sav"

try:
    loaded_model = pk.load(open(model_path, 'rb'))
    
    # Make predictions
    prediction = loaded_model.predict(df)
    prediction_proba = loaded_model.predict_proba(df)

    # Display results
    st.subheader('Predicted Result')
    st.write('Yes, the passenger survived' if prediction[0] == 1 else 'No, the passenger did not survive')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

except FileNotFoundError:
    st.error(f"Model file not found at: {model_path}")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
