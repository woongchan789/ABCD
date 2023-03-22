import streamlit as st
from PIL import Image
from models.My_u2net import remove
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Create a new sketch", page_icon="🍊")

st.image('abcd\\css\\002.png', use_column_width=True)
st.image('abcd\\css\\002_description.png', use_column_width=True)

roberts_1 = np.array([[ 1, 0, 0],
                      [ 0, 0, 0],
                      [ 0, 0,-1]])

roberts_2 = np.array([[ 0, 0, 1],
                      [ 0, 0, 0],
                      [-1, 0, 0]])

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[ 1, 2, 1],
                    [ 0, 0, 0],
                    [-1,-2,-1]])

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[ 1, 1, 1],
                      [ 0, 0, 0],
                      [-1,-1,-1]])

LoG_3_1 = np.array([[ 0,-1, 0],
                    [-1, 4,-1],
                    [ 0,-1, 0]])

LoG_3_2 = np.array([[-1,-1,-1],
                    [-1, 8,-1],
                    [-1,-1,-1]])

img = st.file_uploader('Upload Image ▼', type=['jpg', 'png', 'jpeg'])

if img is None:
    st.text('')
else:
    st.image(img, use_column_width=True)
    image = Image.open(img)
    
    h, w = image.size

    with st.spinner('Wait for it...'):
        
        output = remove(image)

        output = np.array(output)

        gray_frame = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        res1 = cv2.filter2D(gray_frame, -1, roberts_1)
        res2 = cv2.filter2D(gray_frame, -1, roberts_2)
        res3 = cv2.filter2D(gray_frame, -1, sobel_x)
        res4 = cv2.filter2D(gray_frame, -1, sobel_y)
        res5 = cv2.filter2D(gray_frame, -1, prewitt_x)
        res6 = cv2.filter2D(gray_frame, -1, prewitt_y)
        res7 = cv2.filter2D(gray_frame, -1, LoG_3_1)
        res8 = cv2.filter2D(gray_frame, -1, LoG_3_2)

        final_array = []
        after_filtering = np.asarray([res1, res2, res3, res4, res5, res6, res7, res8])

        for i in range(after_filtering.shape[1]):
            for j in range(after_filtering.shape[2]):
                temp = []
                for k in range(after_filtering.shape[0]):
                    temp.append(after_filtering[k][i][j])
                final_array.append(max(temp))

        # black version    
        final_img_black = np.array(final_array).reshape(w, h)
        img_black = Image.fromarray(final_img_black)

        # white version
        array_255 = np.full(len(final_array), 255)
        final = np.abs(final_array - array_255)

        final_img = np.array(final).reshape(w, h)
        plt.imsave('new_sketch.jpeg', final_img, cmap=cm.gray)
        
        img_white = cv2.imread('new_sketch.jpeg')
        time.sleep(1)
        st.success('Done!')

    st.info('   BLACK VERSION', icon="⬛")    
    st.image(img_black, use_column_width=True)
    st.info('   WHITE VERSION', icon="⬜")
    st.image(img_white, use_column_width=True)
    
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
