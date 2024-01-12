import streamlit as st
import time 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import tensorflow as tf

from tensorflow import keras
import keras_cv



def main() :
    try : 
        st.title('Genereador de Imagen por prompt')

        st.write('Escribe lo que deseas generar a traves del modelo generador de imagenes ')
        text_image_generator = st.text_input('Prompt to Create Images', 'Horse on the rainbow')

        model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

        images = model.text_to_image(text_image_generator, batch_size=1)

        plot_images(images)
    except Exception as e :
        print(e)

def plot_images(images):
    for i in range(len(images)):
        st.image(images[i], use_column_width=True)
    plt.close()

if __name__ == '__main__':
    main()