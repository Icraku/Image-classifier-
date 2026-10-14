import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

# from python_utils import LazyCastedDict
# import cv
# from keras.preprocessing import image
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
# from utils import classify

# set title
# st.title('Dohpay Solutions')

# set header
# st.header('Hello!')
# st.write('#### At Dohpay solutions, we are help farmers be informed of maize diseases for them to make informed '
# 'choices when treating the diseases')
# st.text('Please upload a maize leaf image')
# st.text('If the image is from the web, upload URL of the image here')


import base64 # image conversion


#@st.cache_data
#def get_base64_of_bin_file(file):
#    with open(file, 'rb') as f:
#        data = f.read()
#    return base64.b64encode(data).decode()

#img = get_base64_of_bin_file("C:/Users/ADMIN/Desktop/maize_img/bg-image/bg2.jpg")
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''

    st.markdown(
        f"""
         <style>
         .stApp .stHeader{{
             background: url("https://img.freepik.com/premium-photo/field-corn-with-green-background_882954-17435.jpg?w=360");
             background-size: cover
         </style>
         """,
        unsafe_allow_html=True
    )


set_bg_hack_url()

html_temp = """
    <div style='background-color:#f63366;
    padding:10px:margin-bottom:25px'> 
    <h2 
    style='color:white;text-align:center;'>Dohpay Solutions</h2> 
    <p style='color:white;text-align:center;'>At Dohpay solutions, we help farmers know maize disease infesting their crops for them to make informed choices when treating the 
    diseases.</p> 
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)


# upload file
# file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
def load_model():
    model = tf.keras.models.load_model('./saved_model/1')
    return model


with st.spinner('Please wait.....'):
    model = load_model()

# load class names
classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']


# scale image
def scale(image):
    image = tf.cast(image, tf.float32)
    image /= 256.0
    return tf.image.resize(image, [256, 256])


def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = scale(img)
    return np.expand_dims(img, axis=0)

    # the URL option path = st.text_input('Enter image URL if image is from the web...',
    # 'http://barmac.com.au/wp-content/uploads/sites/3/2016/01/Angular-Leaf-Spot') if path is not None: content =
    # requests.get(path).content


# noinspection PyTypeChecker
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


disease_info = {
    'Blight': {
        'Description': 'Blight is a fungal plant disease that primarily affects maize leaves. It appears as dark, '
                       'irregularly shaped lesions with a yellow border. Blight can spread rapidly and have a negative'
                       ' crop yield.',
        'Treatment': 'Treatment includes applying copper-based or sulfur-based fungicides to infected plants. '
                     'Also, consider removing and destroying severely affected plants to prevent further spread.',
    },
    'Common_Rust': {
        'Description': 'Common Rust is a fungal disease that commonly affects maize leaves. It appears as small, round,'
                       ' reddish-brown pustules on the leaf surface. Severe infections can lead to reduced '
                       'photosynthesis and yield loss.',
        'Treatment': 'To control Common Rust, you can use fungicides like triazole or strobilurin fungicides. '
                     'Also consider proper crop rotation and selecting resistant maize varieties to help prevent the '
                     'disease.',
    },
    'Gray_Leaf_Spot': {
        'Description': 'Gray Leaf Spot is a fungal disease that affects leaves of maize plants. It is characterized by'
                       ' rectangular, grayish lesions with yellow halos on the leaves. Gray Leaf Spot can reduce '
                       'photosynthesis and crop yield if not managed properly.',
        'Treatment': 'To properly manage Gray Leaf Spot, apply fungicides like chlorothalonil or azoxystrobin.'
                     ' Ensure there is proper spacing between plants to improve air circulation and reducing '
                     'leaf wetness can also help prevent the disease.',
    },
    'Healthy': {
        'Description': 'Your maize leaves appear to be healthy! Healthy maize leaves should be green, without any '
                       'noticeable lesions or discoloration. Proper care and maintenance are essential to maintain the '
                       'health of your crop.',
        'Treatment': 'No treatment is required for healthy leaves. However, ensure you regularly monitor and use '
                     'appropriate agricultural practices like proper irrigation and fertilization to keep the plants '
                     'healthy.',
    }
}

option = st.radio('', ['Choose a test image', 'Choose your own image'])
if option == 'Choose your own image':
    uploaded_file = st.file_uploader('Choose an image...', type=['jpeg', 'jpg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img_batch = np.expand_dims(img, 0)
        predictions = model.predict(img_batch)
        predicted_class = classes[np.argmax(predictions[0])]
        confidence = np.round(np.max(predictions[0])*100, 2)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, width=200)
        with col2:
            st.success('Disease name: ' + str(predicted_class))
            st.info('Prediction Confidence: ' + str(confidence) + '%')
        st.subheader('Disease details and Possible Treatment solutions:')
        if predicted_class in disease_info:
            st.write('**Description:**\n\n', disease_info[predicted_class]['Description'])
            st.write('**Treatment:**\n\n', disease_info[predicted_class]['Treatment'])
        else:
            st.warning("No information found for this disease.")
else:
    test_images = os.listdir('sample_images')
    test_images = st.selectbox('Please select a test image: ', test_images)
    file_path = 'sample_images/' + test_images
    img = Image.open(file_path).convert('RGB')
    img_batch = np.expand_dims(img, 0)
    predictions = model.predict(img_batch)
    predicted_class = classes[np.argmax(predictions[0])]
    confidence = np.round(np.max(predictions[0]) * 100, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, width=200)
    with col2:
        st.success('Disease name: ' + str(predicted_class))
        st.info('Prediction Confidence: ' + str(confidence) + '%')
    st.subheader('Disease details and Possible Treatment solutions:')
    if predicted_class in disease_info:
        st.write('**Description:**\n\n', disease_info[predicted_class]['Description'])
        st.write('**Treatment:**\n\n', disease_info[predicted_class]['Treatment'])
    else:
        st.warning("No information found for this disease.")
# display image


# write classification
# result = st.button('Classify this image')
# if result:
# st.spinner('Classifying.Please wait....')
# st.write('### The predicted class is: {}'.format(predicted_class))
# st.write('#### The confidence level: {}'.format(float(confidence) * 100, 2), '%')

# image = Image.open(file).convert('RGB')
# st.image(image, caption='Classifying Maize leaf Image', use_column_width=True)
