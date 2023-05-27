import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2
from PIL import Image
import base64

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #748A88;
    }
</style>
""", unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('Background/MNIST.jpg')  

model_architecture = "ModelNhanDangChuSoVietTayMNIST\\digit_config.json"
model_weights = "ModelNhanDangChuSoVietTayMNIST\\digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 

optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]) 

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test_image = X_test

RESHAPED = 784

X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')

#normalize in [0,1]
X_test /= 255

#@st.cache
def create_random_image():
    st.session_state.index = np.random.randint(0, 9999, 150)
    digit_random = np.zeros((10*28, 15*28), dtype=np.uint8)
    for i in range(0, 150):
        m = i // 15
        n = i % 15
        digit_random[m*28:(m+1)*28, n*28:(n+1)*28] = X_test_image[st.session_state.index[i]] 
    cv2.imwrite('ModelNhanDangChuSoVietTayMNIST\\digit_random.jpg', digit_random)
    return 'ModelNhanDangChuSoVietTayMNIST\\digit_random.jpg'


#@st.cache
def recognize_digits(index, image):
    X_test_sample = np.zeros((150, 784), dtype=np.float32)
    for i in range(0, 150):
        X_test_sample[i] = X_test[index[i]] 

    prediction = model.predict(X_test_sample)
    s = ''
    for i in range(0, 150):
        ket_qua = np.argmax(prediction[i])
        
        s = s + str(ket_qua) + ' '
        if (i+1) % 15 == 0:
            s = s + '\n'
            s=f'<div style="font-family:Consolas; color:white; font-size: 25px; ">{s}</div>'
    return s

def main():
	
    st.title('Nhan dang chu so viet tay')
    #st.geometry('520x550')
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'image_path' not in st.session_state:
        st.session_state.image_path = None

    btn_tao_anh = st.button('Tao anh')
   
    if btn_tao_anh:
        image = Image.open(create_random_image())
        st.image(image, width=421)
        result = recognize_digits(st.session_state.index, st.session_state.image_path)
        # st.write(result)
        #st.write(create_random_image())

        st.write('<span style="font-family:Consolas; color:white; font-size: 30px;">Nhận dạng:</span>',unsafe_allow_html=True)
        
        # result=f'<div style="font-family:Consolas; color:black; font-size: 15px;background-color: #83C448; padding: 10px;">{result}</div>'
        # st.markdown(st.text(result), unsafe_allow_html=True)
        # st.text(result)
        st.write(result,unsafe_allow_html=True)


if __name__ == "__main__":
    main()
