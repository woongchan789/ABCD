import streamlit as st

st.set_page_config(
    page_title="ABCD(Any Body Can be a Designer",
    page_icon="🎨",
)

st.image('abcd\\css\\logo.png', use_column_width=True)
st.title('ABCD(Any Body Can be a Designer)')
st.subheader('by woongchan Nam')

st.markdown(
    """
    ABCD(Any Body Can be a Designer) is a platform designed to assist with various graphic tasks.  
    Enjoy easy functions based on AI models and  
    Experience the opportunity to become a painter by creating personalized works!
    ***  
    👈 **Select a funtion and experience it!** 
    ### Four Funtions ABCD Offers
    #### 1. Upload picture and Get the abstract image
    - 이미지를 업로드해서 손쉽게 원하는 일러스트, 픽토그램, 스케치를 다운로드 받으세요!
    #### 2. Create a new sketch
    - 이미지의 배경을 제거한 뒤 line만을 살린 새로운 sketch를 다운로드 받으세요!
    #### 3. Remove background
    - 배경을 제거된 오브젝트를 다운로드 받으세요!
    #### 4. Neural Style transfer
    - 반고흐, 피카소 등 화가의 화풍을 살린 작품을 만들어 보세요!


"""
)

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
