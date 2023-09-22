
import streamlit as st
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def init(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Cargar el modelo
RandomForest =joblib.load("model_g.pkl")
LinearRegression =joblib.load("model_linear.pkl")
DecisionTree =joblib.load("model_decision.pkl")


st.title('Modelo_RandomSearch')

# Asumiendo que el modelo necesita dos características (puedes adaptar esto según tus necesidades)

feature1 = st.number_input('longitude', value=1.0)
feature2 = st.number_input('latitude', value=1.0)
feature3 = st.number_input('housing_median_age', value=1.0)
feature4 = st.number_input('total_rooms', value=1.0)
feature5 = st.number_input('total_bedrooms', value=1.0)
feature6 = st.number_input('population', value=1.0)
feature7 = st.number_input('households', value=1.0)
feature8 = st.number_input('median_income', value=1.0)
feature9 = st.selectbox('ocean_proximity', ('<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'))
modelSelect = st.selectbox("model", ("decision tree", "linear regresion", "random forest"))

# Crear un numpy array con las características
features = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]).reshape(1, -1)

predictionDF = pd.DataFrame.from_dict({"longitude":[feature1], "latitude":[feature2],"housing_median_age":[feature3],"total_rooms":[feature4],"total_bedrooms":[feature5],
                      "population":[feature6],"households":[feature7],"median_income":[feature8],"ocean_proximity":[feature9]})

# Realizar la predicción
if st.button('Realizar Predicción'):
    if modelSelect == "decision tree":
        model = DecisionTree
    elif modelSelect == "linear regresion":
        model = LinearRegression
    elif modelSelect == "random forest":
        model = RandomForest
    prediction = model.predict(predictionDF)
    st.write(f'La predicción es:', {prediction[0]})