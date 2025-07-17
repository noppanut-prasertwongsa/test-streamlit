import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดลที่ฝึกไว้แล้ว
try:
    model = joblib.load('house_price_model.joblib')
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดล 'house_price_model.joblib'. กรุณารันไฟล์ 'train_model.py' ก่อน")
    st.stop()


# สร้าง UI ของเว็บแอป
st.title('House Price Prediction App')
st.write('แอปพลิเคชันทำนายราคาบ้านอย่างง่ายด้วย Linear Regression')

# สร้าง Sidebar สำหรับรับ Input จากผู้ใช้
st.sidebar.header('ใส่ข้อมูลบ้านของคุณ:')

def user_input_features():
    area = st.sidebar.slider('ขนาดพื้นที่ (ตร.ม.)', 50, 300, 100)
    bedrooms = st.sidebar.slider('จำนวนห้องนอน', 1, 6, 3)
    bathrooms = st.sidebar.slider('จำนวนห้องน้ำ', 1, 4, 2)
    age = st.sidebar.slider('อายุบ้าน (ปี)', 0, 50, 5)
    
    data = {'Area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Age': age}
    
    features = pd.DataFrame(data, index=[0])
    return features

# รับข้อมูลจากผู้ใช้
input_df = user_input_features()

# แสดงข้อมูลที่ผู้ใช้ป้อน
st.subheader('ข้อมูลที่คุณป้อน:')
st.write(input_df)

# ปุ่มสำหรับทำนาย
if st.button('ทำนายราคา'):
    # ทำนายราคา
    prediction = model.predict(input_df)
    
    # แสดงผลการทำนาย
    st.subheader('ผลการทำนาย:')
    st.success(f'ราคาบ้านที่ทำนายได้คือ: {prediction[0]:,.2f} บาท')