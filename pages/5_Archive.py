from PIL import Image
import pandas as pd
import streamlit as st
import os
import math

st.set_page_config(page_title="Archive", page_icon="📁")

st.header("📂 Archive storage")

label_df = pd.read_csv('abcd\\label_list.csv', encoding='cp949')

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 0.5, .2, 0.5, .2, 0.5, .2, 0.5, .2))
with row2_1:
    total = "🎨 " + "Total 420,310 files"
    st.markdown(total)
with row2_2:
    illust = "🔴 " + "illust 83,466 files"
    st.markdown(illust)
with row2_3:
    pictogram = "🟠 " + "pictogram 128,172 files"
    st.markdown(pictogram)
with row2_4:
    sketch = "🟡 " + "sketch 208,672 files"
    st.markdown(sketch)

st.subheader('STEP1. Please check the class name first')
see_data = st.expander('You can click here to see the label list first 👉')
with see_data:
    st.markdown('1️⃣ There are a total of 1006 classes.')
    st.dataframe(data=label_df.reset_index(drop=True), use_container_width=True)

    st.markdown('2️⃣ Please select a category level1')
    level1 = st.selectbox("",
    (label_df.level1.unique()), label_visibility='collapsed')
    st.markdown('')

    st.markdown('3️⃣ Please select a category level2')
    level2 = st.selectbox("",
    (label_df[label_df['level1'] == level1].level2.unique()), label_visibility='collapsed')
    st.markdown('4️⃣ If you have a class name you want and **Please remember it!**')
    st.markdown('')
    st.dataframe(data=label_df[(label_df['level1'] == level1) & (label_df['level2'] == level2)].level3.unique(), use_container_width=True)

st.subheader('STEP2. Choose the type you want')
option = st.selectbox("",
    ('illust', 'pictogram', 'sketch'), label_visibility='collapsed')
st.markdown('')

st.subheader('STEP3. Choose the class you have checked')
final_level1 = st.selectbox("",
    (label_df.level1.unique()), label_visibility='collapsed', key="final_level1")
st.markdown('')

final_level2 = st.selectbox("",
    (label_df[label_df['level1'] == final_level1].level2.unique()), label_visibility='collapsed', key="final_level2")
st.markdown('')

final_level3 = st.selectbox("",
    (label_df[label_df['level2'] == final_level2].level3.unique()), label_visibility='collapsed', key="final_level3")
st.markdown('')

if option=='illust':
    root_dir = 'D:\\abcd\\project\\data\\20211217_153350\\image_data\\1.Training\\ABSTRACT_ILLUSTRATION'
    want_class_number = int(label_df[label_df['level3'] == final_level3]['id'])
    destination = root_dir + str('/') + str(want_class_number)
    file_list = os.listdir(destination)

    i = 0
    for file in file_list:
        path = destination + str('/') + file
        img = Image.open(path)
        if (i % 3) == 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img)
        elif (i % 3) == 1:    
            with col2:
                st.image(img)
        else:    
            with col3:
                st.image(img)
        i += 1

if option=='pictogram':
    root_dir = 'D:\\abcd\\project\\data\\20211217_153350\\image_data\\1.Training\\ABSTRACT_PICTOGRAM'
    want_class_number = int(label_df[label_df['level3'] == final_level3]['id'])
    destination = root_dir + str('/') + str(want_class_number)
    file_list = os.listdir(destination)

    i = 0
    for file in file_list:
        path = destination + str('/') + file
        img = Image.open(path)
        if (i % 3) == 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img)
        elif (i % 3) == 1:    
            with col2:
                st.image(img)
        else:    
            with col3:
                st.image(img)
        i += 1

if option=='sketch':
    root_dir = 'D:\\abcd\\project\\data\\20211217_153350\\image_data\\1.Training\\ABSTRACT_SKETCH'
    want_class_number = int(label_df[label_df['level3'] == final_level3]['id'])
    destination = root_dir + str('/') + str(want_class_number)
    file_list = os.listdir(destination)

    i = 0
    for file in file_list:
        path = destination + str('/') + file
        img = Image.open(path)
        if (i % 3) == 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img)
        elif (i % 3) == 1:    
            with col2:
                st.image(img)
        else:    
            with col3:
                st.image(img)
        i += 1

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
