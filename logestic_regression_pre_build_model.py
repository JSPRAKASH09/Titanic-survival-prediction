#-*- coding : utf-8-*-
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
# from pickel import dump
# from pickel import load 
import pickle as pk
import warnings
warnings.simplefilter("ignore")

st.title('Model Deployment : Logestic regression')
st.sidebar.header('Claiminants Information')
def user_input_features():
	CLMSEX = st.sidebar.selectbox('Gender', ('1', '0')) 
	CLMINSUR= st.sidebar.selectbox('Insurance', ('1','0')) 
	SEATBELT= st.sidebar.selectbox('SeatBelt', (' 1', '0')) 
	CLMAGE= st.sidebar.number_input("Insert the Age") 
	LOSS= st.sidebar.number_input("Insert Loss") 
	data={'CLMSEX': CLMSEX, 
		'CLMINSUR' :CLMINSUR, 
		'SEATBELT':SEATBELT, 
		'CLMAGE': CLMAGE, 
		'LOSS': LOSS} 
	features= pd.DataFrame(data, index=[0]) 
	return features 
df =user_input_features() 
st.subheader('User Input parameters') 
st.write(df) 
#load the model from disk 
# loaded_model= pk.load(open(r"C:\Users\pjspr\OneDrive\Data Science\Data_science_assignments\Completed\Logistic Regression\logestic_regression.sav", 'rb'))
file_path = r"C:\Users\pjspr\OneDrive\Data Science\Data_science_assignments\Completed\Logistic Regression\logistic_regression.sav"

with open(file_path, "rb") as file:
    loaded_model = pk.load(file)

# print(loaded_model.feature_names_in_)
prediction= loaded_model.predict(df)
print(prediction)
prediction_proba = loaded_model.predict_proba(df) 
st.subheader('Predicted Result') 
st.write('Yes, Claiminant appiont an attorney' if prediction_proba[0][1] <0.5 else 'No, the claiminant will not appoint an attorney') 
st.subheader('Prediction Probability') 
st.write(prediction_proba) 
