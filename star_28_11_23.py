import streamlit as st
from joblib import load
import numpy as np

# Loading model
model_lr = load('finalized_model.sav')
model_rf = load('best_random_forest_model.sav')
model_svm = load('best_svm_model.sav')

# Tu código Streamlit aquí...


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
url_imagen_fondo = "https://www.nasa.gov/sites/default/files/thumbnails/image/stsci-h-p2041a-f-2073x1382.png"

st.markdown(f"""
    <style>
    .reportview-container {{
        background: url("{url_imagen_fondo}");
        background-size: cover;
    }}
    .big-font {{
        font-size:30px !important;
        color: #FF4B4B;
    }}
    .title-font {{
        font-size:50px !important;
        color: #4CAF50;
        font-weight: bold;
    }}
    .stSelectbox, .stButton > button {{
        width: 100%;
        border-radius: 5px;
        border: 2px solid #4CAF50;
    }}
    .stButton > button {{
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        height: 3em;
    }}
    </style>
    """, unsafe_allow_html=True)

# Título de la aplicación
st.markdown('<p class="title-font">Star Classification Predictive Model</p>', unsafe_allow_html=True)

features = feature_input()

class_map = {
    0: "GALAXY",
    1: "QSO",
    2: "STAR"
}


# Make a prediction and display it
# Selector para elegir el modelo
# Selector de modelo
st.markdown('<p class="big-font">Elige un modelo para la predicción:</p>', unsafe_allow_html=True)
model_option = st.selectbox(' ', ('Random Forest', 'Logistic Regression', 'SVM'))

# Lógica para elegir el modelo basado en la selección del usuario
if model_option == 'Random Forest':
    model = model_rf
elif model_option == 'Logistic Regression':
    model = model_lr
else:
    model = model_svm



if st.button('Predecir'):
    prediction = model.predict(features)
    st.write(f'El objeto astronómico es: {class_map[prediction[0]]}')


