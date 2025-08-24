import streamlit as st
from PIL import Image
import pandas as pd
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
#------------------------------------------------------------------------------------------------
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer, AddMissingIndicator
from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.pipeline import Pipeline
# import input.preprocessors as pp  # Asegúrate de que este módulo esté disponible
from configuraciones import config  # Asegúrate de que este módulo esté disponible
#------------------------------------------------------------------------------------------------

def prediccion_o_inferencia(pipeline_de_test, datos_de_test):
    try:
        datos_de_test['battery_time'] = datos_de_test['battery_time'].astype('O')
        datos_de_test = datos_de_test[config.FEATURES]

        new_vars_with_na = [
            var for var in config.FEATURES
            if var not in config.CATEGORICAL_VARS_WITH_NA_FREQUENT +
            config.CATEGORICAL_VARS_WITH_NA_MISSING +
            config.NUMERICAL_VARS_WITH_NA
            and datos_de_test[var].isnull().sum() > 0
        ]

        datos_de_test.dropna(subset=new_vars_with_na, inplace=True)
        predicciones = pipeline_de_test.predict(datos_de_test)
        predicciones_sin_escalar = np.exp(predicciones)
        return predicciones, predicciones_sin_escalar, datos_de_test
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        return None, None, None

#Diseno de la Interface
st.title("Proyecto Modelo ML - Miguel Franco Bayas - DATAPATH")

image = Image.open('src/images/datapath-logo.png') #src/
st.image(image, use_container_width=True) #use_column_width esta "deprecated"

st.sidebar.write("Suba el archivo CSV correspondiente para realizar la predicción")

#------------------------------------------------------------------------------------------
# Cargar el archivo CSV desde la barra lateral
uploaded_file = st.sidebar.file_uploader(" ", type=['csv'])

if uploaded_file is not None:
    #Leer el archivo CSV y lo pasamos a Dataframe
    df_de_los_datos_subidos = pd.read_csv(uploaded_file)

    #Mostrar el contenido del archivo CSV
    st.write('Contenido del archivo CSV en formato Dataframe:')
    st.dataframe(df_de_los_datos_subidos)
#-------------------------------------------------------------------------------------------
# Cargar el Modelo ML o Cargar el Pipeline
pipeline_path = os.path.join(os.path.dirname(__file__), 'linear_regression.joblib')

# Verificar si el modelo existe
if os.path.exists(pipeline_path):
    try:
        pipeline_de_produccion = joblib.load(pipeline_path)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        pipeline_de_produccion = None
else:
    st.error(f"No se encontró el modelo en la ruta: {pipeline_path}")
    pipeline_de_produccion = None

if st.sidebar.button("Haz clic aquí para enviar el CSV al Pipeline"):
    if uploaded_file is None:
        st.sidebar.write("No se cargó correctamente el archivo, súbalo de nuevo")
    elif pipeline_de_produccion is None:
        st.sidebar.write("No se cargó correctamente el modelo")
    else:
        with st.spinner('Pipeline y Modelo procesando...'):
            prediccion, prediccion_sin_escalar, datos_procesados = prediccion_o_inferencia(pipeline_de_produccion, df_de_los_datos_subidos)
            time.sleep(2)
            st.success('¡Listo!')

            if prediccion is not None:
                # Mostrar los resultados de la predicción
                st.write('Resultados de la predicción:')
                st.write(prediccion)
                st.write(prediccion_sin_escalar)

                # Graficar los precios de venta predichos
                fig, ax = plt.subplots()
                pd.Series(prediccion_sin_escalar).hist(bins=50, ax=ax)
                ax.set_title('Histograma de rango de precios de móviles')
                ax.set_xlabel('Precio')
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)

                # Proceso para descargar todo el archivo con las predicciones
                df_resultado = datos_procesados.copy()
                df_resultado['Predicción Escalada'] = prediccion
                df_resultado['Predicción Sin Escalar'] = prediccion_sin_escalar

                # Mostrar el Dataframe concatenado
                st.write('Datos originales con predicciones:')
                st.dataframe(df_resultado)

                # Crear el archivo CSV para descargar
                csv = df_resultado.to_csv(index=False).encode('utf-8')

                # Botón para descargar el CSV
                st.download_button(
                    label="Descargar archivo CSV con predicciones",
                    data=csv,
                    file_name='predicciones_moviles.csv',
                    mime='text/csv',
                )
            #-------------------------------------------------------------------
#Comando para lanzar la aplicación de forma LOCAL:
#streamlit run modelo_ml_streamlit.py
