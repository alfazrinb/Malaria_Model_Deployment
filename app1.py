
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

model = tf.keras.models.load_model('my_model.hdf5')

st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction
         """
         )

st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a Maculopathy")
    elif np.argmax(prediction) == 1:
        st.write("It is a Normal")
    elif np.argmax(prediction) == 2:
        st.write("it ia a Pathological Myopia")
    else:
        st.write("It is a Rhegmatogenous RD!")
    
    st.text("Probability (0: Maculopathy, 1: Normal, 2: Pathological Myopia, 3: Rhegmatogenous RD!)")
    st.write(prediction)
