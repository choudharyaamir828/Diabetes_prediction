import numpy as np
import pickle 
import streamlit as st

#loaded the saved model
loaded_model = pickle.load(open('trained_model.pkl','rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
   

    # changing the input data to numpy array
    input_data_as_numpy_array = np.array(input_data)

    # reshape the array as we are Preducting for one instance because we only we 
    # one input not give whole data 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'the person is not diabetic'
    else:
        return 'the person is diabetic '

def main():

    # giving title
    st.title('DIabetes prediction web app')

    # getting the input data from the user
    pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('SkinThickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction value')
    Age = st.text_input('Age of the persion')

    #code for prediction 
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([ pregnancies, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
    
