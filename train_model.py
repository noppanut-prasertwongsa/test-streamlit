import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# 1. สร้างข้อมูลจำลอง (Dummy Data)
# สร้างข้อมูลให้มีความสัมพันธ์กันเล็กน้อยเพื่อให้โมเดลเรียนรู้ได้
np.random.seed(42) # เพื่อให้ได้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
data = {
    'Area': np.random.randint(800, 3500, 20),
    'Bedrooms': np.random.randint(1, 6, 20),
    'Bathrooms': np.random.randint(1, 4, 20),
    'Age': np.random.randint(0, 50, 20)
}
df = pd.DataFrame(data)

# สร้างราคาบ้าน (Target variable) โดยให้มีความสัมพันธ์กับ features อื่นๆ
# ราคา = (Area * 100) + (Bedrooms * 5000) - (Age * 200) + สุ่มค่ารบกวน
df['Price'] = (df['Area'] * 100 + 
               df['Bedrooms'] * 5000 - 
               df['Age'] * 200 + 
               np.random.randint(-10000, 10000, 20))

print("ข้อมูลตัวอย่างที่สร้างขึ้น:")
print(df)

# 2. เตรียมข้อมูลและสอนโมเดล
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']] # Features
y = df['Price'] # Target

# สร้างและสอนโมเดล Linear Regression
model = LinearRegression()
model.fit(X, y)

print("\nโมเดลได้ถูกสอนเรียบร้อยแล้ว")

# 3. บันทึกโมเดล
# ใช้ joblib ในการบันทึกโมเดล
joblib.dump(model, 'house_price_model.joblib')

print("โมเดลถูกบันทึกเป็นไฟล์ 'house_price_model.joblib' เรียบร้อยแล้ว")