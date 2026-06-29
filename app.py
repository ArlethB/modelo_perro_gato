import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- CONFIGURACIÓN ----------------

st.set_page_config(
    page_title="Clasificador de Perros y Gatos",
    page_icon="🐶🐱",
    layout="centered"
)

st.title("🐶🐱 Clasificador de Perros y Gatos")

st.write("**Nombre del estudiante:** Arleth Bonilla")
st.write("**Número de cuenta:** 20221900251")

st.write(
    "Sube una imagen y el modelo de Inteligencia Artificial indicará si corresponde a un perro o a un gato."
)

IMG_SIZE = (224, 224)

# Carpeta donde están el modelo y las clases
MODEL_DIR = Path("modelo_perros_gatos")

CLASS_PATH = MODEL_DIR / "class_names.json"

MODEL_PATHS = [
    MODEL_DIR / "modelo_perros_gatos.keras",
    MODEL_DIR / "modelo_perros_gatos.h5",
]

# ---------------- CARGAR MODELO ----------------

@st.cache_resource
def cargar_modelo():

    for ruta in MODEL_PATHS:
        if ruta.exists():
            return tf.keras.models.load_model(ruta, compile=False)

    st.error("No se encontró el modelo.")
    st.stop()


@st.cache_data
def cargar_clases():

    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as archivo:
            return json.load(archivo)

    return ["gatos", "perros"]


modelo = cargar_modelo()
clases = cargar_clases()

# ---------------- PREPROCESAMIENTO ----------------

def preparar_imagen(imagen):

    imagen = imagen.convert("RGB")
    imagen = imagen.resize(IMG_SIZE)

    arreglo = np.array(imagen, dtype=np.float32)

    arreglo = tf.keras.applications.mobilenet_v2.preprocess_input(arreglo)

    return np.expand_dims(arreglo, axis=0)


# ---------------- PREDICCIÓN ----------------

def predecir(imagen):

    predicciones = modelo.predict(
        preparar_imagen(imagen),
        verbose=0
    )[0]

    indice = np.argmax(predicciones)

    clase = clases[indice]

    confianza = predicciones[indice] * 100

    return clase, confianza, predicciones


# ---------------- INTERFAZ ----------------

archivo = st.file_uploader(
    "Seleccione una imagen",
    type=["jpg", "jpeg", "png"]
)

if archivo:

    imagen = Image.open(archivo)

    st.image(
        imagen,
        caption="Imagen seleccionada",
        use_container_width=True
    )

    clase, confianza, predicciones = predecir(imagen)

    st.subheader("Resultado")

    st.success(
        f"Predicción: **{clase.capitalize()}** ({confianza:.2f}%)"
    )

    st.subheader("Probabilidades")

    for nombre, probabilidad in zip(clases, predicciones):

        st.write(
            f"**{nombre.capitalize()}**: {probabilidad*100:.2f}%"
        )

else:

    st.info("Seleccione una imagen para comenzar.")
