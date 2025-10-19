import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# ----------------- CONFIGURACIÓN DE PÁGINA -----------------
st.set_page_config(
    page_title="Clasificación de Imágenes",
    page_icon="🧠",
    layout="wide"
)

# ----------------- ESTILOS -----------------
st.markdown("""
    <style>
    .title {text-align: center; font-size: 38px; font-weight: bold; color: #2c3e50;}
    .subtitle {text-align: center; font-size: 18px; color: #7f8c8d; margin-bottom: 20px;}
    .footer {text-align: center; font-size: 13px; color: #95a5a6; margin-top: 40px;}
    .prediction {font-size: 20px; font-weight: bold; color: #16a085;}
    </style>
""", unsafe_allow_html=True)

# ----------------- INFORMACIÓN DEL SISTEMA -----------------
st.sidebar.markdown("### ⚙️ Información del sistema")
st.sidebar.write("Versión de Python:", platform.python_version())

# ----------------- CARGA DEL MODELO -----------------
with st.spinner("Cargando modelo entrenado..."):
    model = load_model('keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# ----------------- ENCABEZADO -----------------
st.markdown("<p class='title'>🖼️ Reconocimiento de Imágenes</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Clasifica tus fotos en categorías usando un modelo entrenado en Teachable Machine</p>", unsafe_allow_html=True)

# Imagen de referencia (logo/ejemplo)
image = Image.open('OIG5.jpg')
st.image(image, width=350, caption="Imagen de ejemplo")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.subheader("ℹ️ Acerca de esta app")
    st.info("Puedes usar un modelo entrenado en **Teachable Machine** para identificar imágenes capturadas con tu cámara.")

# ----------------- CAPTURA DE IMAGEN -----------------
st.markdown("### 📸 Toma una foto con tu cámara")
img_file_buffer = st.camera_input("Haz clic en el botón para capturar")

# ----------------- PROCESO DE PREDICCIÓN -----------------
if img_file_buffer is not None:
    # Inicializar arreglo de datos
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Leer la imagen como PIL
    img = Image.open(img_file_buffer)

    # Redimensionar a 224x224
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir a array
    img_array = np.array(img)

    # Normalizar
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    # Cargar en data
    data[0] = normalized_image_array

    # Inferencia
    with st.spinner("🔎 Analizando imagen..."):
        prediction = model.predict(data)

    # ----------------- RESULTADOS -----------------
    st.markdown("### 📊 Resultados de la predicción")
    if prediction[0][0] > 0.5:
        st.success(f"👉 **Arriba** con probabilidad: {prediction[0][0]:.2f}")
    if prediction[0][1] > 0.5:
        st.success(f"👉 **Abajo** con probabilidad: {prediction[0][1]:.2f}")
    # if prediction[0][2] > 0.5:
    #     st.success(f"👉 **Derecha** con probabilidad: {prediction[0][2]:.2f}")

# ----------------- PIE DE PÁGINA -----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>🚀 Aplicación desarrollada con Streamlit + Keras | Modelo de Teachable Machine</p>", unsafe_allow_html=True)


