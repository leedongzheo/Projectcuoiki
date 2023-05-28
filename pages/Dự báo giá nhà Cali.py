import joblib
import streamlit as st
import pandas as pd
import numpy as np
import base64
def my_format(x):
    s = "{:,.0f}".format(x)
    L = len(s)
    if L < 14:
        s = '&nbsp'*(14-L) + s
    return s
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
add_bg_from_local('Background/california.png')  
st.markdown("""
    <style>
    .scrollable-table {
        max-width: 100%;
        max-height: 500px;
        overflow-x: auto;
        overflow-y: auto;
    }
    table {
     
        color: blue;
        background-color: lightpink;
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
    }

    th, td {
        border: 1px solid darkgray;
        padding: 8px;
        text-align: left;
    }

    th {
        background-color: #2DC1DB;
        color: black;
        position: sticky;
        top: 0;
    }
    </style>
""", unsafe_allow_html=True)

forest_reg = joblib.load("ModelDuBaoGiaNhaCaLi\\forest_reg_model.pkl")

column_names=['longitude','latitude','housing_median_age','total_rooms',
              'total_bedrooms','population','households','median_income',
              'rooms_per_household','population_per_household',
              'bedrooms_per_room','ocean_proximity_1', 
              'ocean_proximity_2', 'ocean_proximity_3', 
              'ocean_proximity_4', 'ocean_proximity_5']
st.subheader('Dự báo giá nhà California')
x_test = pd.read_csv('ModelDuBaoGiaNhaCaLi\\x_test.csv', header = None, names=column_names)
y_test = pd.read_csv('ModelDuBaoGiaNhaCaLi\\y_test.csv', header = None)
y_test = y_test.to_numpy()
N = len(x_test)
st.write('<div class="scrollable-table">',x_test.to_html(escape=False, index=False), unsafe_allow_html=True)


get_5_rows = st.button('Lấy 5 hàng ngẫu nhiên và dự báo')
if get_5_rows:
    index = np.random.randint(0,N-1,5)
    some_data = x_test.iloc[index]
    st.write('<div class="scrollable-table">',some_data.to_html(escape=False, index=False), unsafe_allow_html=True)
    result = 'y_test:' + '&nbsp&nbsp&nbsp&nbsp' 
    for i in index:
        s = my_format(y_test[i,0])
        result = result + s
    result = f'<div style="font-family:Consolas; color:black; font-size: 15px;background-color: #83C448; padding: 10px;">{result}</div>'
    st.markdown(result, unsafe_allow_html=True)

    some_data = some_data.to_numpy()
    y_pred = forest_reg.predict(some_data)
    result = 'y_predict:' + '&nbsp'
    for i in range(0, 5):
        s = my_format(y_pred[i])
        result = result + s
    result=f'<div style="font-family:Consolas; color:black; font-size: 15px;background-color: #83C448; padding: 10px;">{result}</div>'
    st.markdown(result, unsafe_allow_html=True)

