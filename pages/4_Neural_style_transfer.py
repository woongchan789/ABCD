import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import time

st.set_page_config(page_title="Neural style transfer", page_icon="🍈")

st.image('abcd\\css\\004.png', use_column_width=True)
st.image('abcd\\css\\004_description.png', use_column_width=True)

style_transfer_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def perform_style_transfer(content_image, style_image):
    content_image = tf.convert_to_tensor(content_image, np.float32)[tf.newaxis, ...] / 255.
    style_image = tf.convert_to_tensor(style_image, np.float32)[tf.newaxis, ...] / 255.
    
    output = style_transfer_model(content_image, style_image)
    stylized_image = output[0]
    
    return Image.fromarray(np.uint8(stylized_image[0] * 255))

content_image_input = st.file_uploader('Upload Content Image ▼', type=['jpg'])
if content_image_input is None:
    st.text('')
else:
    st.image(content_image_input, use_column_width=True)
    style_image_input = st.file_uploader('Upload Style Image ▼', type=['jpg'])
    if style_image_input is None:
        st.text('')
    else:
        st.image(style_image_input, use_column_width=True)
        content_img = Image.open(content_image_input)
        style_img = Image.open(style_image_input)

        content_img = np.array(content_img)
        style_img = np.array(style_img)
        final_img = perform_style_transfer(content_img, style_img)
        with st.spinner('Wait for it...'):
            time.sleep(13)
        st.success('Done!')
        st.image(final_img, use_column_width=True)

#Add a feedback section in the sidebar
st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.subheader('Please help us improve!')
with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    rating=st.slider("Please rate the app", min_value=1, max_value=5, value=3,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
    type=st.radio("What's type", ('😥 Misclassification', '🐔 Front-end', '🐸 Back-end', '🐞 Etc'), label_visibility='collapsed')
    text=st.text_input(label='Please leave your feedback here')
    submitted = st.form_submit_button('Submit')
    if submitted:
      st.write('Thanks for your feedback!')
      st.markdown('Your Rating:')
      st.markdown(rating)
      st.markdown('Your Feedback:')
      st.markdown(text)    
