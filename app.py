import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import cv2
import os 
import gdown

def download_model():
    file_id = "11SP3nB63G5a0F_RRaK5Dy2r-n3Jtu-nf"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "Brain_Tumors_Classifier.keras"
    if not os.path.exists(output_path):
        st.info("Downloading model from Google Drive...")
        gdown.download(url, output_path, quiet=False)
        st.success("Model downloaded!")

# حمل الموديل
download_model()
model = load_model('Brain_Tumors_Classifier.keras')


class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


st.title("Brain Tumor Classification")
st.write("Upload a brain MRI image to detect the tumor type.")


uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # زر التنبؤ
    if st.button("Predict"):
        # تحويل الصورة إلى NumPy Array
        img = np.array(image.resize((224, 224)))

        # معالجة القنوات
        if img.ndim == 2 or img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[-1] == 4:
            img = img[:, :, :3]

        img = np.expand_dims(img, axis=0)

        # التنبؤ
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]

        # عرض النتيجة
        st.success(f"Prediction: **{predicted_class}**")
