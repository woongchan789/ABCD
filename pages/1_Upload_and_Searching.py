from models.My_resnet50 import resnet50
import streamlit as st
import time
from PIL import Image
import pandas as pd
import os
import torch
import sys
from torchvision import transforms 
import torch.nn as nn

sys.path.insert(0, 'models')

st.set_page_config(page_title="Upload picture and Get the abstract image", page_icon="🍎")

st.image('abcd\\css\\001.png', use_column_width=True)
st.image('abcd\\css\\001_description.png', use_column_width=True)


label_df = pd.read_csv('abcd\\label_list.csv', encoding='cp949')
kor_df = pd.read_excel('abcd\\label_eng_to_kor_translation.xlsx')
cls_df = pd.read_csv('abcd\\wrong_classes.csv', encoding='cp949')

file = st.file_uploader('Upload Image ▼', type=['jpg', 'png'])

if file is None:
    st.text('')
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1006)
    model.load_state_dict(torch.load('weights\\resnet50_model.pt'))

    totensor = transforms.ToTensor()
    image = totensor(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        pred = model(image)
        _, index = torch.max(pred,1)

    final_class = int(index) + 1
    kor_final_class = label_df[label_df['id'] == final_class].iloc[0,3]

    with st.spinner('Wait for it...'):
        time.sleep(1)

    question = 'You want a image of ' + str(kor_final_class) + '?'
    st.info(question, icon='🤔')

    correct = st.button('✅ Correct')
    incorrect = st.button('❌ Incorrect')

    if correct:
        st.markdown('---')
        st.balloons()
        result = "You want an abstract image of " + str(kor_final_class) + "!"
        st.info(result, icon='👌')
        st.success('Choose the type you want (illust, pictogram, sketch)!', icon="😋")
        illust_expand = st.expander('🔴 ILLUSTRATION 👉')
        with illust_expand:
            st.markdown('')
            root_dir = 'abcd\\project\\data\\20211217_153350\\image_data\\1.Training\\ABSTRACT_ILLUSTRATION'
            destination = root_dir + str('\\') + str(final_class)
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

        pictogram_expand = st.expander('🟠 PICTOGRAM 👉')
        with pictogram_expand:
            st.markdown('')                
            root_dir = 'abcd\\project\\data\\20211217_153350\\image_data\\1.Training\\ABSTRACT_PICTOGRAM'
            destination = root_dir + str('\\') + str(final_class)
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
        sketch_expand = st.expander('🟡 SKETCH 👉')
        with sketch_expand:
            st.markdown('')
            root_dir = 'abcd\\project\\data\\20211217_153350\\image_data\\1.Training\\ABSTRACT_SKETCH'
            destination = root_dir + str('\\') + str(final_class)
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

    if incorrect:
        st.markdown('---')
        st.error("Sorry. Please check the archive storage and search for the desired class")
        st.error("👈 We would appreciate it if you could leave the misclassification in the feedback section so that it can be improved")

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
