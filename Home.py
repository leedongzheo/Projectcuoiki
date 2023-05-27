import streamlit as st
import base64

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("<h1 style='font-size: 50px;color:black'>Welcome to the website of Nguyen Tran Anh Thu! 👋</h1>", unsafe_allow_html=True)
st.sidebar.success("You can choose one of my projects above.")


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
add_bg_from_local('Background/Home.jpg')  
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #748A88;
    }
</style>
""", unsafe_allow_html=True)
  
st.markdown(
    """
    <style>
    .red-text {
        color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="red-text">
        Hello! Thank you for visiting my website. Here are the projects I have learned through the Machine Learning and Computer Vision courses taught by Mr. Tran Tien Duc.
        This website is created for the purpose of reporting the final projects for these two courses and is only used for educational purposes.
        👉 You can choose one of the projects I have learned on the left side.
         <p><b style="font-size: 40px;">Personal Information:</b></p>
        <div>
	- Full name: Nguyen Tran Anh Thu</p>
        - Student code: 21146157</p>
        - School name: HCMC University of Technology and Education
        </div>
	<p><b style="font-size: 40px;">Contact me:</b></p>
	<div>
        <p>- Github: <a style="color:green" href="https://github.com/leedongzheo">https://github.com/leedongzheo</a></p>
        <p>- Facebook:<a style="color:green" href="https://www.facebook.com/anhthu.nguyentran.142035"> https://www.facebook.com/anhthu.nguyentran.142035</a></p>
        <p>- Email: <a style="color:green" href="leedongzheo@gmail.com"> leedongzheo@gmail.com</a></p>
        <p>- Phone: <a style="color:green" href="0916870014"> 0916870014</a></p>
	</div>
	<p><b style="font-size: 40px;">Instructor Information:</b></p>
   	<div>
        <p>- Teacher: Tran Tien Duc</p>
        <p>- Email:<a style="color:green" href="ductt@hcmute.edu.vn"> ductt@hcmute.edu.vn</a></p>
        <p>- Phone:<a style="color:green" href="ductt@hcmute.edu.vn"> 0919622862</a></p>
        <p>- Github: <a style="color:green" href="https://github.com/TranTienDuc">https://github.com/TranTienDuc</a><p>
	</div>
    </div>
    """,
    unsafe_allow_html=True
)