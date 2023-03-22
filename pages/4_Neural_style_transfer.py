import streamlit as st
from PIL import Image
from models.neural_style_transfer_model import run_style_transfer

import time
import torch
from models import My_vgg19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Neural style transfer", page_icon="🍈")

st.image('abcd\\css\\004.png', use_column_width=True)
st.image('abcd\\css\\004_description.png', use_column_width=True)

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
        cnn = My_vgg19.VGG19().features.to(device).eval()
        with st.spinner('Wait for it... It takes about 10 seconds.'):
            final_img = run_style_transfer(cnn, style_img=style_img, content_img=content_img)
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
