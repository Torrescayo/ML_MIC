import streamlit as st
from joblib import load
import numpy as np

# Loading model
model = load('/home/torrescayo/ML/finalized_model.sav')

def feature_input():
    u = st.number_input('Ultraviolet filter')
    g = st.number_input('green filter')
    r = st.number_input('red filter')
    i = st.number_input('Near ifrared filter')
    z = st.number_input('infrared filter(z)')
    spec_obj_ID = st.number_input('Spec object')
    redshift = st.number_input('redshift value')
    plate = st.number_input('plate')
    MJD = st.number_input('MJD')
    return np.array([[u, g, r, i, z, spec_obj_ID, redshift, plate, MJD]]).reshape(1, -1)

# Set up the app
st.title('Star Classification Predictive Model')

features = feature_input()

class_map = {
    0: "GALAXY",
    1: "QSO",
    2: "STAR"
}


# Make a prediction and display it
if st.button('Predecir'):
    prediction = model.predict(features)
    st.write(f'El objeto astronómico es: {class_map[prediction[0]]}')


