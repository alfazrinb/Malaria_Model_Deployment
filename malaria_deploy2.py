
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('malaria_model1.hdf5')

st.write("""
         # Malaria Virus Platform Prediction
         """
         )

st.write("This is a simple image classification web app to predict Malaria Virus")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a Falciparum")
    elif np.argmax(prediction) == 1:
        st.write("It is a Malariae")
    elif np.argmax(prediction) == 2:
        st.write("it ia a Ovale")
    else:
        st.write("It is a Vivax")
    
    st.text("Probability (0: Falciparum, 1: Malariae, 2: Ovale, 3: Vivax)")
    st.write(prediction)
