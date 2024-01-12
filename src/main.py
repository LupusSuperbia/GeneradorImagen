import streamlit as st
import time 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import tensorflow as tf
from tensorflow import keras
import keras_cv

# Decorador para caché
@st.cache_data(allow_output_mutation=True)
def cargar_modelo():
    # Crear y cargar el modelo
    return keras_cv.models.StableDiffusion(img_width=512, img_height=512)

def main():
    try:
        st.title('Generador de Imagen por Prompt')

        st.write('Escribe lo que deseas generar a través del modelo generador de imágenes')
        text_image_generator = st.text_input('Prompt to Create Images', 'Horse on the rainbow')

        # Cargar el modelo desde la caché
        model = cargar_modelo()

        with st.spinner("Generando imágenes..."):
            images = model.text_to_image(text_image_generator, batch_size=1)

        plot_images(images)
    except Exception as e:
        st.error(f"Error: {e}")

def plot_images(images):
    for i in range(len(images)):
        st.image(images[i], use_column_width=True)
    plt.close()

if __name__ == '__main__':
    main()
