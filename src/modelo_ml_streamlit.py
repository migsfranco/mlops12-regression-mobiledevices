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
from configuraciones import config

#------------------------------------------------------------------------------------------------
def prediccion_o_inferencia(pipeline_de_test, datos_de_test):
    try:
        # Obtener las columnas disponibles en el DataFrame
        available_columns = datos_de_test.columns.tolist()

        # Filtrar las columnas esperadas que están disponibles en el DataFrame
        features_available = [col for col in config.FEATURES if col in available_columns]

        if not features_available:
            st.error("No hay columnas coincidentes entre el archivo CSV y las columnas esperadas.")
            return None, None, None

        datos_de_test = datos_de_test[features_available]

        # Verificar si 'battery_time' está en las columnas disponibles
        if 'battery_time' in datos_de_test.columns:
            datos_de_test['battery_time'] = datos_de_test['battery_time'].astype('O')

        new_vars_with_na = [
            var for var in features_available
            if var not in config.CATEGORICAL_VARS_WITH_NA_FREQUENT +
            config.CATEGORICAL_VARS_WITH_NA_MISSING +
            config.NUMERICAL_VARS_WITH_NA
            and datos_de_test[var].isnull().sum() > 0
        ]

        datos_de_test.dropna(subset=new_vars_with_na, inplace=True)

        # Verificar si el DataFrame no está vacío después de eliminar filas con NaN
        if datos_de_test.empty:
            st.error("El DataFrame está vacío después de eliminar filas con valores NaN.")
            return None, None, None

        predicciones = pipeline_de_test.predict(datos_de_test)
        predicciones_sin_escalar = np.exp(predicciones)
        return predicciones, predicciones_sin_escalar, datos_de_test
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        return None, None, None

# Diseño de la Interfaz
st.title("Proyecto Modelo ML - Miguel Franco Bayas - DATAPATH")

# Ruta de la imagen
image_path = os.path.join(os.path.dirname(__file__), 'images/datapath-logo.png')

# Verificar si la imagen existe
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
else:
    st.error(f"No se encontró la imagen en la ruta: {image_path}")

st.sidebar.write("Suba el archivo CSV correspondiente para realizar la predicción")

#------------------------------------------------------------------------------------------
# Cargar el archivo CSV desde la barra lateral
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        df_de_los_datos_subidos = pd.read_csv(uploaded_file)
        st.write('Contenido del archivo CSV en formato Dataframe:')
        st.dataframe(df_de_los_datos_subidos)

        # Validar columnas disponibles
        available_columns = df_de_los_datos_subidos.columns.tolist()
        missing_columns = [col for col in config.FEATURES if col not in available_columns]
        if missing_columns:
            st.warning(f"Las siguientes columnas no están en el archivo CSV y serán ignoradas: {missing_columns}")
        else:
            st.write('Todas las columnas necesarias están presentes.')
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {e}")

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
                st.subheader('Resultados de la Predicción')
                st.write('Predicción Escalada (log-precio):')
                st.write(prediccion)
                st.write('Predicción Sin Escalar (precio en euros):')
                st.write(prediccion_sin_escalar)

                # Graficar los precios
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(prediccion_sin_escalar, bins=30, color='skyblue', edgecolor='black')
                ax.set_title('Distribución de Precios Predichos')
                ax.set_xlabel('Precio en Euros')
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)

                # Proceso para descargar todo el archivo con las predicciones
                df_resultado = df_de_los_datos_subidos.copy()
                df_resultado['prediccion_log_precio'] = prediccion
                df_resultado['prediccion_precio_euros'] = prediccion_sin_escalar

                # Mostrar el Dataframe concatenado
                st.subheader('Dataframe con las Predicciones')
                st.dataframe(df_resultado)

                # Crear el archivo CSV para descargar
                csv = df_resultado.to_csv(index=False).encode('utf-8')

                # Botón para descargar el CSV
                st.download_button(
                    label="Descargar archivo CSV con las Predicciones",
                    data=csv,
                    file_name='predicciones_modelo_ml.csv',
                    mime='text/csv',
                )