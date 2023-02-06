from rembg import remove
import streamlit as st
from PIL import Image
import time

st.set_page_config(page_title="Remove background", page_icon="🍋")

st.image('abcd\\css\\003.png', use_column_width=True)
st.image('abcd\\css\\003_description.png', use_column_width=True)

file = st.file_uploader('Upload image ▼', type=['jpg', 'png'])

if file is None:
    st.text('')
else:
    st.image(file, use_column_width=True)
    image = Image.open(file)
    output = remove(image)
    with st.spinner('Wait for it...'):
        time.sleep(10)
    st.success('Done!')
    st.image(output, use_column_width=True)

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
