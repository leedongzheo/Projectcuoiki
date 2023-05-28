import streamlit as st
import math
import base64
st.set_page_config(page_title="Giải phương trình bậc 2", page_icon="📌")
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
add_bg_from_local('Background/Quadratic Equation Solver.png')  

st.subheader("Giải phương trình bậc 2")
st.sidebar.header("Giải phương trình bậc 2")

with st.form(key = 'my-form'):
    a = st.number_input('Nhập a:')
    b = st.number_input('Nhập b:')
    c = st.number_input('Nhập c:')
    submit_button_giai = st.form_submit_button(label='Giải')

if submit_button_giai:
    if a == 0.0:
        if b == 0.0:
            if c == 0.0:
                ket_qua = 'PTB1 vô số nghiệm'
            else:
                ket_qua = 'PTB1 vô nghiệm'
        else:
            x = -c/b
            ket_qua = 'PTB1 có nghiệm x = %.2f' % x
    else:
        delta = b**2 - 4*a*c
        if delta < 0:
            ket_qua = 'PTB2 vô nghiệm'
        else:
            x1 = (-b + math.sqrt(delta))/(2*a)
            x2 = (-b - math.sqrt(delta))/(2*a)
            # ket_qua = "<span style='color:red;'>PTB2 có nghiệm x1 = %.2f và x2 = %.2f</span>" % (x1, x2)
            ket_qua = 'PTB2 có nghiệm x1 = %.2f và x2 = %.2f' % (x1, x2)
    st.write(f"<span style='color:red; font-size:50px;'>{ket_qua}</span>", unsafe_allow_html=True)


