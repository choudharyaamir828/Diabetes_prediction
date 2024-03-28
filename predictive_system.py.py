import pickle
import streamlit as st
import numpy as np

# liading the saved model 
loaded_model = pickle.load(open('C:/Users/AMIR/OneDrive/Desktop/preduction/trained_model.sav','rb'))

input_data = (3, 110, 70, 30, 80, 27.5, 0.45, 35)

# changing the input data to numpy array
input_data_as_numpy_array = np.array(input_data)

# reshape the array as we are Preducting for one instance because we only we 
# one input not give whole data 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Standardize the input data 
#std_data = scaler.transform(input_data_reshaped)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction == 0:
    print("the person is not diabetic")
else:
    print('the person is diabetic ')